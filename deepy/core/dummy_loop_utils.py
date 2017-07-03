#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
The code in this file is from the scan function in theano.
Never modify this file directly.
"""


import logging
import numpy
import warnings

from theano.compat import ifilter, izip
from six import iteritems, integer_types
from six.moves import xrange
from theano.compile import SharedVariable, function
from theano import compile
from theano import gof
from theano.tensor import opt
from theano import tensor
from theano import config
from theano.updates import OrderedUpdates
from theano.compile import ops
from theano.compat import OrderedDict


from theano.scan_module import scan_op
from theano.scan_module import scan_utils
from theano.scan_module.scan_utils import safe_new, traverse

# Logging function for sending warning or info
_logger = logging.getLogger('scan_dummy_args')



def get_dummy_args(sequences=None,
         outputs_info=None,
         non_sequences=None,
         n_steps=None,
         truncate_gradient=-1,
         go_backwards=False,
         mode=None,
         name=None,
         profile=False,
         allow_gc=None,
         strict=False):
    ################################################################## P1>
    # check if inputs are just single variables instead of lists
    def wrap_into_list(x):
        """
        Wrap the input into a list if it is not already a list.

        """
        if x is None:
            return []
        elif not isinstance(x, (list, tuple)):
            return [x]
        else:
            return list(x)

    seqs = wrap_into_list(sequences)
    outs_info = wrap_into_list(outputs_info)

    # Make sure we get rid of numpy arrays or ints or anything like that
    # passed as inputs to scan
    non_seqs = []
    for elem in wrap_into_list(non_sequences):
        if not isinstance(elem, gof.Variable):
            non_seqs.append(tensor.as_tensor_variable(elem))
        else:
            non_seqs.append(elem)

    # If we provided a known number of steps ( before compilation)
    # and if that number is 1 or -1, then we can skip the Scan Op,
    # and just apply the inner function once
    # To do that we check here to see the nature of n_steps
    n_fixed_steps = None

    if isinstance(n_steps, (float, integer_types)):
        n_fixed_steps = int(n_steps)
    else:
        try:
            n_fixed_steps = opt.get_scalar_constant_value(n_steps)
        except tensor.basic.NotScalarConstantError:
            n_fixed_steps = None

    # Check n_steps is an int
    if (hasattr(n_steps, 'dtype') and
        str(n_steps.dtype)[:3] not in ('uin', 'int')):
        raise ValueError(' n_steps must be an int. dtype provided '
                         'is %s' % n_steps.dtype)

    # compute number of sequences and number of outputs
    n_seqs = len(seqs)
    n_outs = len(outs_info)

    return_steps = OrderedDict()
    # wrap sequences in a dictionary if they are not already dictionaries
    for i in xrange(n_seqs):
        if not isinstance(seqs[i], dict):
            seqs[i] = OrderedDict([('input', seqs[i]), ('taps', [0])])
        elif seqs[i].get('taps', None) is not None:
            seqs[i]['taps'] = wrap_into_list(seqs[i]['taps'])
        elif seqs[i].get('taps', None) is None:
            # seqs dictionary does not have the ``taps`` key
            seqs[i]['taps'] = [0]

    # wrap outputs info in a dictionary if they are not already in one
    for i in xrange(n_outs):
        if outs_info[i] is not None:
            if isinstance(outs_info[i], dict):
                # DEPRECATED :
                if outs_info[i].get('return_steps', None) is not None:
                    raise ValueError(
                            "Using `return_steps` has been deprecated. "
                            "Simply select the entries you need using a "
                            "subtensor. Scan will optimize memory "
                            "consumption, so do not worry about that.")
                # END

            if not isinstance(outs_info[i], dict):
                # by default any output has a tap value of -1
                outs_info[i] = OrderedDict([('initial', outs_info[i]), ('taps', [-1])])
            elif (outs_info[i].get('initial', None) is None and
                    outs_info[i].get('taps', None) is not None):
                # ^ no initial state but taps provided
                raise ValueError(('If you are using slices of an output '
                                  'you need to provide a initial state '
                                  'for it'), outs_info[i])
            elif (outs_info[i].get('initial', None) is not None and
                  outs_info[i].get('taps', None) is None):
                # ^ initial state but taps not provided
                if 'taps' in outs_info[i]:
                    # ^ explicitly provided a None for taps
                    _logger.warning('Output %s ( index %d) has a initial '
                            'state but taps is explicitly set to None ',
                             getattr(outs_info[i]['initial'], 'name', 'None'),
                             i)
                outs_info[i]['taps'] = [-1]
        else:
            # if a None is provided as the output info we replace it
            # with an empty OrdereDict() to simplify handling
            outs_info[i] = OrderedDict()

    ##
    # Step 2. Generate inputs and outputs of the inner functions
    # for compiling a dummy function (Iteration #1)
    ##

    # create theano inputs for the recursive function
    # note : this is a first batch of possible inputs that will
    #        be compiled in a dummy function; we used this dummy
    #        function to detect shared variables and their updates
    #        and to construct a new and complete list of inputs and
    #        outputs

    n_seqs = 0
    scan_seqs = []     # Variables passed as inputs to the scan op
    inner_seqs = []    # Variables passed as inputs to the inner function
    inner_slices = []  # Actual slices if scan is removed from the picture
    # go through sequences picking up time slices as needed
    for i, seq in enumerate(seqs):
        # Note that you can have something like no taps for
        # a sequence, though is highly unlikely in practice
        if 'taps' in seq:
            # go through the indicated slice
            mintap = numpy.min(seq['taps'])
            maxtap = numpy.max(seq['taps'])
            for k in seq['taps']:
                # create one slice of the input
                # Later on, if we decide not to use scan because we are
                # going for just one step, it makes things easier if we
                # compute the correct outputs here. This way we can use
                # the output of the lambda expression directly to replace
                # the output of scan.

                # If not we need to use copies, that will be replaced at
                # each frame by the corresponding slice
                actual_slice = seq['input'][k - mintap]
                _seq_val = tensor.as_tensor_variable(seq['input'])
                _seq_val_slice = _seq_val[k - mintap]
                nw_slice = _seq_val_slice.type()

                # Try to transfer test_value to the new variable
                if config.compute_test_value != 'off':
                    try:
                        nw_slice.tag.test_value = gof.Op._get_test_value(
                            _seq_val_slice)
                    except AttributeError as e:
                        if config.compute_test_value != 'ignore':
                            # No need to print a warning or raise an error now,
                            # it will be done when fn will be called.
                            _logger.info(('Cannot compute test value for '
                                'the inner function of scan, input value '
                                'missing %s'), e)

                # Add names to slices for debugging and pretty printing ..
                # that is if the input already has a name
                if getattr(seq['input'], 'name', None) is not None:
                    if k > 0:
                        nw_name = seq['input'].name + '[t+%d]' % k
                    elif k == 0:
                        nw_name = seq['input'].name + '[t]'
                    else:
                        nw_name = seq['input'].name + '[t%d]' % k
                    nw_slice.name = nw_name

                # We cut the sequence such that seq[i] to correspond to
                # seq[i-k]. For the purposes of cutting the sequences, we
                # need to pretend tap 0 is used to avoid cutting the sequences
                # too long if the taps are all lower or all higher than 0.
                maxtap_proxy = max(maxtap, 0)
                mintap_proxy = min(mintap, 0)
                start = (k - mintap_proxy)
                if k == maxtap_proxy:
                    nw_seq = seq['input'][start:]
                else:
                    end = -(maxtap_proxy - k)
                    nw_seq = seq['input'][start:end]

                if go_backwards:
                    nw_seq = nw_seq[::-1]

                scan_seqs.append(nw_seq)
                inner_seqs.append(nw_slice)
                inner_slices.append(actual_slice)
                n_seqs += 1

    # Since we've added all sequences now we need to level them up based on
    # n_steps or their different shapes
    lengths_vec = []
    for seq in scan_seqs:
        lengths_vec.append(seq.shape[0])

    if not scan_utils.isNaN_or_Inf_or_None(n_steps):
        # ^ N_steps should also be considered
        lengths_vec.append(tensor.as_tensor(n_steps))

    if len(lengths_vec) == 0:
        # ^ No information about the number of steps
        raise ValueError('No information about the number of steps '
                         'provided. Either provide a value for '
                         'n_steps argument of scan or provide an input '
                         'sequence')

    # If the user has provided the number of steps, do that regardless ( and
    # raise an error if the sequences are not long enough )
    if scan_utils.isNaN_or_Inf_or_None(n_steps):
        actual_n_steps = lengths_vec[0]
        for contestant in lengths_vec[1:]:
            actual_n_steps = tensor.minimum(actual_n_steps, contestant)
    else:
        actual_n_steps = tensor.as_tensor(n_steps)

    # Add names -- it helps a lot when debugging

    for (nw_seq, seq) in zip(scan_seqs, seqs):
        if getattr(seq['input'], 'name', None) is not None:
            nw_seq.name = seq['input'].name + '[%d:]' % k

    scan_seqs = [seq[:actual_n_steps] for seq in scan_seqs]
    # Conventions :
    #   mit_mot = multiple input taps, multiple output taps ( only provided
    #             by the gradient function )
    #   mit_sot = multiple input taps, single output tap (t + 0)
    #   sit_sot = single input tap, single output tap (t + 0)
    #   nit_sot = no input tap, single output tap (t + 0)

    # MIT_MOT -- not provided by the user only by the grad function
    n_mit_mot = 0
    n_mit_mot_outs = 0
    mit_mot_scan_inputs = []
    mit_mot_inner_inputs = []
    mit_mot_inner_outputs = []
    mit_mot_out_slices = []
    mit_mot_rightOrder = []

    # SIT_SOT -- provided by the user
    n_mit_sot = 0
    mit_sot_scan_inputs = []
    mit_sot_inner_inputs = []
    mit_sot_inner_slices = []
    mit_sot_inner_outputs = []
    mit_sot_return_steps = OrderedDict()
    mit_sot_tap_array = []
    mit_sot_rightOrder = []

    n_sit_sot = 0
    sit_sot_scan_inputs = []
    sit_sot_inner_inputs = []
    sit_sot_inner_slices = []
    sit_sot_inner_outputs = []
    sit_sot_return_steps = OrderedDict()
    sit_sot_rightOrder = []

    # go through outputs picking up time slices as needed
    for i, init_out in enumerate(outs_info):
        # Note that our convention dictates that if an output uses
        # just the previous time step, as a initial state we will only
        # provide a tensor of the same dimension as one time step; This
        # makes code much cleaner for those who do not use taps. Otherwise
        # they would always had to shape_padleft the initial state ..
        # which is ugly
        if init_out.get('taps', None) == [-1]:

            actual_arg = init_out['initial']
            if not isinstance(actual_arg, tensor.Variable):
                actual_arg = tensor.as_tensor_variable(actual_arg)
            arg = safe_new(actual_arg)
            if isinstance(arg, tensor.Constant):
                # safe new returns a clone of the constants, but that is not
                # what we need for initial states
                arg = arg.type()

            # Try to transfer test_value to the new variable
            if config.compute_test_value != 'off':
                try:
                    arg.tag.test_value = gof.Op._get_test_value(actual_arg)
                except AttributeError as e:
                    if config.compute_test_value != 'ignore':
                        # No need to print a warning or raise an error now,
                        # it will be done when fn will be called.
                        _logger.info(('Cannot compute test value for the '
                            'inner function of scan, input value missing %s'),
                                     e)

            if getattr(init_out['initial'], 'name', None) is not None:
                arg.name = init_out['initial'].name + '[t-1]'

            # We need now to allocate space for storing the output and copy
            # the initial state over. We do this using the expand function
            # defined in scan utils
            sit_sot_scan_inputs.append(
                scan_utils.expand_empty(
                    tensor.unbroadcast(
                        tensor.shape_padleft(actual_arg), 0),
                    actual_n_steps
                ))

            sit_sot_inner_slices.append(actual_arg)
            if i in return_steps:
                sit_sot_return_steps[n_sit_sot] = return_steps[i]
            sit_sot_inner_inputs.append(arg)
            sit_sot_rightOrder.append(i)
            n_sit_sot += 1

        elif init_out.get('taps', None):

            if numpy.any(numpy.array(init_out.get('taps', [])) > 0):
                # Make sure we do not have requests for future values of a
                # sequence we can not provide such values
                raise ValueError('Can not use future taps of outputs',
                                    init_out)
            # go through the taps
            mintap = abs(numpy.min(init_out['taps']))
            mit_sot_tap_array.append(init_out['taps'])
            idx_offset = abs(numpy.min(init_out['taps']))
            # Sequence
            mit_sot_scan_inputs.append(
                scan_utils.expand_empty(init_out['initial'][:mintap],
                                        actual_n_steps))

            if i in return_steps:
                mit_sot_return_steps[n_mit_sot] = return_steps[i]
            mit_sot_rightOrder.append(i)
            n_mit_sot += 1
            for k in init_out['taps']:
                # create a new slice
                actual_nw_slice = init_out['initial'][k + mintap]
                _init_out_var = tensor.as_tensor_variable(init_out['initial'])
                _init_out_var_slice = _init_out_var[k + mintap]
                nw_slice = _init_out_var_slice.type()

                # Try to transfer test_value to the new variable
                if config.compute_test_value != 'off':
                    try:
                        nw_slice.tag.test_value = gof.Op._get_test_value(
                            _init_out_var_slice)
                    except AttributeError as e:
                        if config.compute_test_value != 'ignore':
                            # No need to print a warning or raise an error now,
                            # it will be done when fn will be called.
                            _logger.info(('Cannot compute test value for '
                                'the inner function of scan, input value '
                                'missing. %s'), e)

                # give it a name or debugging and pretty printing
                if getattr(init_out['initial'], 'name', None) is not None:
                    if k > 0:
                        nw_slice.name = (init_out['initial'].name +
                                            '[t+%d]' % k)
                    elif k == 0:
                        nw_slice.name = init_out['initial'].name + '[t]'
                    else:
                        nw_slice.name = (init_out['initial'].name +
                                            '[t%d]' % k)
                mit_sot_inner_inputs.append(nw_slice)
                mit_sot_inner_slices.append(actual_nw_slice)
        # NOTE: there is another case, in which we do not want to provide
        #      any previous value of the output to the inner function (i.e.
        #      a map); in that case we do not have to do anything ..

    # Re-order args
    max_mit_sot = numpy.max([-1] + mit_sot_rightOrder) + 1
    max_sit_sot = numpy.max([-1] + sit_sot_rightOrder) + 1
    n_elems = numpy.max([max_mit_sot, max_sit_sot])
    _ordered_args = [[] for x in xrange(n_elems)]
    offset = 0
    for idx in xrange(n_mit_sot):
        n_inputs = len(mit_sot_tap_array[idx])
        if n_fixed_steps in [1, -1]:
            _ordered_args[mit_sot_rightOrder[idx]] = \
                            mit_sot_inner_slices[offset:offset + n_inputs]
        else:
            _ordered_args[mit_sot_rightOrder[idx]] = \
                            mit_sot_inner_inputs[offset:offset + n_inputs]
        offset += n_inputs

    for idx in xrange(n_sit_sot):
        if n_fixed_steps in [1, -1]:
            _ordered_args[sit_sot_rightOrder[idx]] = \
                                        [sit_sot_inner_slices[idx]]
        else:
            _ordered_args[sit_sot_rightOrder[idx]] = \
                                        [sit_sot_inner_inputs[idx]]

    ordered_args = []
    for ls in _ordered_args:
        ordered_args += ls
    if n_fixed_steps in [1, -1]:
        args = (inner_slices +
                ordered_args +
                non_seqs)

    else:
        args = (inner_seqs +
                ordered_args +
                non_seqs)

    # add only the non-shared variables and non-constants to the arguments of
    # the dummy function [ a function should not get shared variables or
    # constants as input ]
    dummy_args = [arg for arg in args
                  if (not isinstance(arg, SharedVariable) and
                      not isinstance(arg, tensor.Constant))]
    ################################################################## P1<
    return dummy_args, locals()


def finish_scan(fn_outputs, local_vars):

    n_fixed_steps = local_vars["n_fixed_steps"]
    return_steps = local_vars["return_steps"]
    non_seqs = local_vars["non_seqs"]
    dummy_args = local_vars["dummy_args"]
    args = local_vars["args"]
    outs_info = local_vars["outs_info"]
    n_outs = local_vars["n_outs"]
    mit_sot_inner_outputs = local_vars["mit_sot_inner_outputs"]
    sit_sot_inner_outputs = local_vars["sit_sot_inner_outputs"]
    sit_sot_scan_inputs = local_vars["sit_sot_scan_inputs"]
    sit_sot_inner_inputs = local_vars["sit_sot_inner_inputs"]
    actual_n_steps = local_vars["actual_n_steps"]
    sit_sot_rightOrder = local_vars["sit_sot_rightOrder"]
    strict = local_vars["strict"]
    non_sequences = local_vars["non_sequences"]
    inner_seqs = local_vars["inner_seqs"]
    mit_mot_inner_inputs = local_vars["mit_mot_inner_inputs"]
    mit_sot_inner_inputs = local_vars["mit_sot_inner_inputs"]
    mit_mot_inner_outputs = local_vars["mit_mot_inner_outputs"]
    mit_sot_tap_array = local_vars["mit_sot_tap_array"]
    allow_gc = local_vars["allow_gc"]
    n_seqs = local_vars["n_seqs"]
    n_mit_mot_outs = local_vars["n_mit_mot_outs"]
    mit_mot_out_slices = local_vars["mit_mot_out_slices"]
    truncate_gradient = local_vars["truncate_gradient"]
    name = local_vars["name"]
    mode = local_vars["mode"]
    profile = local_vars["profile"]
    scan_seqs = local_vars["scan_seqs"]
    mit_mot_scan_inputs = local_vars["mit_mot_scan_inputs"]
    mit_sot_scan_inputs = local_vars["mit_sot_scan_inputs"]
    n_mit_mot = local_vars["n_mit_mot"]
    mit_sot_return_steps = local_vars["mit_sot_return_steps"]
    n_mit_sot = local_vars["n_mit_sot"]
    sit_sot_return_steps = local_vars["sit_sot_return_steps"]
    mit_sot_rightOrder = local_vars["mit_sot_rightOrder"]

    condition, outputs, updates = scan_utils.get_updates_and_outputs(fn_outputs)
    ################################################################## P2>
    if condition is not None:
        as_while = True
    else:
        as_while = False
    ##
    # Step 3. Check if we actually need scan and remove it if we don't
    ##

    if n_fixed_steps in [1, -1]:
        # We do not need to use the scan op anymore, so we can just return
        # the outputs and updates we have
        if condition is not None:
            _logger.warning(('When the number of steps is fixed and equal '
                    'to 1, the provided stopping condition, ',
                    str(condition), ' is ignored'))

        for pos, inner_out in enumerate(outputs):
            # we need to see if we need to pad our sequences with an
            # unbroadcastable dimension; case example : we return an
            # output for which we want all intermediate. If n_steps is 1
            # then, if we return the output as given by the innner function
            # this will represent only a slice and it will have one
            # dimension less.
            if (isinstance(inner_out.type, tensor.TensorType) and
                return_steps.get(pos, 0) != 1):
                outputs[pos] = tensor.unbroadcast(
                    tensor.shape_padleft(inner_out), 0)
        if len(outputs) == 1:
            outputs = outputs[0]

        return (outputs, updates)

    ##
    # Step 4. Compile the dummy function
    ##

    # We can now compile a dummy function just to see what shared variable
    # we have and what are their update rules (note that the user has
    # the option not to pass the shared variable to scan, so we need to
    # pick them manually and add them to scan)
    # make the compilation as fast as possible by not applying any
    # optimization or conversion to C [ note this region is not important
    # for performance so we can do stuff as unoptimal as we wish ]

    # extract still missing inputs (there still might be so) and add them
    # as non sequences at the end of our args
    fake_nonseqs = [x.type() for x in non_seqs]
    fake_outputs = scan_utils.clone(outputs,
                                    replace=OrderedDict(izip(non_seqs,
                                                             fake_nonseqs)))
    all_inputs = ifilter(
        lambda x: (isinstance(x, gof.Variable) and
                   not isinstance(x, SharedVariable) and
                   not isinstance(x, gof.Constant)),
        gof.graph.inputs(fake_outputs))
    extra_inputs = [x for x in all_inputs if x not in args + fake_nonseqs]
    non_seqs += extra_inputs
    # Note we do not use all_inputs directly since the order of variables
    # in args is quite important
    dummy_args += extra_inputs

    dummy_outs = outputs
    if condition is not None:
        dummy_outs.append(condition)
    try:
        dummy_f = function(dummy_args,
                           dummy_outs,
                           updates=updates,
                           mode=compile.mode.Mode(linker='py',
                                                  optimizer=None),
                           on_unused_input='ignore',
                           profile=False)
    except gof.fg.MissingInputError as err:
        msg = ("\nPlease pass this variable to the scan's inner function. Do "
               "not forget to also pass it to the `non_sequences` attribute "
               "of scan.")
        raise gof.fg.MissingInputError(err.args[0] + msg)

    ##
    # Step 5. Re-arange inputs of scan into a more strict order
    ##

    # Step 5.0 Check the outputs of the dummy function to see if they
    # match with user provided data

    # if the number of outputs to the function does not match the number of
    # assumed outputs until now (provided by the user) there can be
    # only one explanation: No information is provided for any of the
    # outputs (i.e. we are dealing with a map)
    tmp_dummy_f_outs = len(dummy_f.maker.outputs)
    if as_while:
        tmp_dummy_f_outs -= 1
    if not (tmp_dummy_f_outs == n_outs or outs_info == []):
        raise ValueError('Please provide None as outputs_info for '
                         'any output that does not feed back into '
                         'scan (i.e. it behaves like a map) ')

    if outs_info == []:
        n_outs = len(dummy_f.maker.outputs)
        if as_while:
            n_outs = n_outs - 1
        outs_info = [OrderedDict() for x in xrange(n_outs)]

    # Step 5.1 Outputs with taps different then -1

    for i, out in enumerate(outs_info):
        if 'taps' in out and out['taps'] != [-1]:
            mit_sot_inner_outputs.append(outputs[i])

    # Step 5.2 Outputs with tap equal to -1
    for i, out in enumerate(outs_info):
        if 'taps' in out and out['taps'] == [-1]:
            sit_sot_inner_outputs.append(outputs[i])

    # Step 5.3 Outputs that correspond to update rules of shared variables
    givens = OrderedDict()
    n_shared_outs = 0
    shared_scan_inputs = []
    shared_inner_inputs = []
    shared_inner_outputs = []
    sit_sot_shared = []
    for input in dummy_f.maker.expanded_inputs:
        if isinstance(input.variable, SharedVariable) and input.update:
            new_var = safe_new(input.variable)
            if getattr(input.variable, 'name', None) is not None:
                new_var.name = input.variable.name + '_copy'
            if isinstance(new_var.type, ops.expandable_types):
                sit_sot_inner_inputs.append(new_var)
                sit_sot_scan_inputs.append(
                    scan_utils.expand_empty(
                        tensor.unbroadcast(
                            tensor.shape_padleft(input.variable), 0),
                        actual_n_steps))
                tensor_update = tensor.as_tensor_variable(input.update)
                sit_sot_inner_outputs.append(tensor_update)
                # Not that pos is not a negative index. The sign of pos is used
                # as a flag to indicate if this output should be part of the
                # update rules or part of the standard outputs of scan.
                # If `pos` is positive than it corresponds to the standard
                # outputs of scan and it refers to output of index `pos`. If `pos`
                # is negative that it corresponds to update rules of scan and it
                # refers to update rule of index -1 - `pos`.
                sit_sot_rightOrder.append(-1 - len(sit_sot_shared))
                sit_sot_shared.append(input.variable)
                givens[input.variable] = new_var

            else:
                shared_inner_inputs.append(new_var)
                shared_scan_inputs.append(input.variable)
                shared_inner_outputs.append(input.update)
                givens[input.variable] = new_var
                n_shared_outs += 1
    n_sit_sot = len(sit_sot_inner_inputs)
    # Step 5.4 Outputs with no taps used in the input
    n_nit_sot = 0
    nit_sot_inner_outputs = []
    nit_sot_return_steps = OrderedDict()
    nit_sot_rightOrder = []
    for i, out in enumerate(outs_info):
        if not 'taps' in out:
            nit_sot_inner_outputs.append(outputs[i])
            if i in return_steps:
                nit_sot_return_steps[n_nit_sot] = return_steps[i]
            nit_sot_rightOrder.append(i)
            n_nit_sot += 1

    # Step 5.5 all other arguments including extra inputs
    other_scan_args = []
    other_inner_args = []

    other_scan_args += [arg for arg in non_seqs
                        if (not isinstance(arg, SharedVariable) and
                            not isinstance(arg, tensor.Constant))]

    # Step 5.6 all shared variables with no update rules
    other_inner_args += [safe_new(arg, '_copy') for arg in non_seqs
                         if (not isinstance(arg, SharedVariable) and
                             not isinstance(arg, tensor.Constant))]

    givens.update(OrderedDict(izip(other_scan_args, other_inner_args)))

    if strict:
        non_seqs_set = set(non_sequences if non_sequences is not None else [])

        other_shared_scan_args = [arg.variable for arg
                            in dummy_f.maker.expanded_inputs
                            if (isinstance(arg.variable, SharedVariable) and
                                not arg.update and
                                arg.variable in non_seqs_set)]
        other_shared_inner_args = [safe_new(arg.variable, '_copy') for arg
                            in dummy_f.maker.expanded_inputs
                            if (isinstance(arg.variable, SharedVariable) and
                                not arg.update and
                                arg.variable in non_seqs_set)]
    else:
        other_shared_scan_args = [arg.variable for arg
                            in dummy_f.maker.expanded_inputs
                            if (isinstance(arg.variable, SharedVariable) and
                                not arg.update)]
        other_shared_inner_args = [safe_new(arg.variable, '_copy') for arg
                            in dummy_f.maker.expanded_inputs
                            if (isinstance(arg.variable, SharedVariable) and
                                not arg.update)]
    givens.update(OrderedDict(izip(other_shared_scan_args,
                                   other_shared_inner_args)))

    ##
    # Step 6. Re-order the outputs and clone them replacing things
    # using the givens
    ##
    inner_inputs = (inner_seqs +
                    mit_mot_inner_inputs +
                    mit_sot_inner_inputs +
                    sit_sot_inner_inputs +
                    shared_inner_inputs +
                    other_shared_inner_args +
                    other_inner_args)

    inner_outs = (mit_mot_inner_outputs +
                  mit_sot_inner_outputs +
                  sit_sot_inner_outputs +
                  nit_sot_inner_outputs +
                  shared_inner_outputs)
    if condition is not None:
        inner_outs.append(condition)
    # Cuda and Gpuarray are imported here, instead of being imported on top of
    # the file because that would force on the user some dependencies that we
    # might do not want to. Currently we are working on removing the
    # dependencies on sandbox code completeley.
    from theano import gpuarray
    if gpuarray.pygpu_activated:
        # very often we end up in this situation when we want to
        # replace w with w_copy, where w is a GPU variable
        # and w_copy is TensorType. This is caused because shared
        # variables are put on GPU right away >:| ,
        new_givens = OrderedDict()
    
        for w, w_copy in iteritems(givens):
            if (isinstance(w.type, gpuarray.GpuArrayType) and
                    isinstance(w_copy.type, tensor.TensorType)):
                for o in inner_outs:
                    new_givens = traverse(o, w, w_copy, new_givens)
            else:
                new_givens[w] = w_copy
    else:
        new_givens = givens

    new_outs = scan_utils.clone(inner_outs, replace=new_givens)

    ##
    # Step 7. Create the Scan Op
    ##

    tap_array = mit_sot_tap_array + [[-1] for x in xrange(n_sit_sot)]
    if allow_gc is None:
        allow_gc = config.scan.allow_gc
    info = OrderedDict()

    info['tap_array'] = tap_array
    info['n_seqs'] = n_seqs
    info['n_mit_mot'] = n_mit_mot
    info['n_mit_mot_outs'] = n_mit_mot_outs
    info['mit_mot_out_slices'] = mit_mot_out_slices
    info['n_mit_sot'] = n_mit_sot
    info['n_sit_sot'] = n_sit_sot
    info['n_shared_outs'] = n_shared_outs
    info['n_nit_sot'] = n_nit_sot
    info['truncate_gradient'] = truncate_gradient
    info['name'] = name
    info['mode'] = mode
    info['destroy_map'] = OrderedDict()
    info['gpua'] = False
    info['as_while'] = as_while
    info['profile'] = profile
    info['allow_gc'] = allow_gc
    info['strict'] = strict

    local_op = scan_op.Scan(inner_inputs, new_outs, info)

    ##
    # Step 8. Compute the outputs using the scan op
    ##
    _scan_inputs = (scan_seqs +
                    mit_mot_scan_inputs +
                    mit_sot_scan_inputs +
                    sit_sot_scan_inputs +
                    shared_scan_inputs +
                    [actual_n_steps for x in xrange(n_nit_sot)] +
                    other_shared_scan_args +
                    other_scan_args)

    scan_inputs = []
    for arg in [actual_n_steps] + _scan_inputs:
        try:
            arg = tensor.as_tensor_variable(arg)
        except TypeError:
            # This happens for Random States for e.g. but it is a good way
            # to make sure no input is a cuda ndarrays
            pass
        scan_inputs += [arg]
    scan_outs = local_op(*scan_inputs)
    if type(scan_outs) not in (list, tuple):
        scan_outs = [scan_outs]
    ##
    # Step 9. Figure out which outs are update rules for shared variables
    # and so on ...
    ##

    update_map = OrderedUpdates()

    def remove_dimensions(outs, steps_return, offsets=None):
        out_ls = []
        for idx, out in enumerate(outs):
            if idx in steps_return:
                if steps_return[idx] > 1:
                    out_ls.append(out[-steps_return[idx]:])
                else:
                    out_ls.append(out[-1])
            else:
                if offsets is None:
                    out_ls.append(out)
                else:
                    out_ls.append(out[offsets[idx]:])
        return out_ls

    offset = n_mit_mot
    offsets = [abs(numpy.min(x)) for x in mit_sot_tap_array]
    mit_sot_outs = remove_dimensions(
        scan_outs[offset:offset + n_mit_sot],
        mit_sot_return_steps,
        offsets)

    offset += n_mit_sot
    offsets = [1 for x in xrange(n_sit_sot)]
    sit_sot_outs = remove_dimensions(
        scan_outs[offset:offset + n_sit_sot],
        sit_sot_return_steps,
        offsets)

    offset += n_sit_sot
    nit_sot_outs = remove_dimensions(
        scan_outs[offset:offset + n_nit_sot],
        nit_sot_return_steps)

    offset += n_nit_sot
    for idx, update_rule in enumerate(
                scan_outs[offset:offset + n_shared_outs]):
        update_map[shared_scan_inputs[idx]] = update_rule

    _scan_out_list = (mit_sot_outs +
                      sit_sot_outs +
                      nit_sot_outs)
    # Step 10. I need to reorder the outputs to be in the order expected by
    # the user
    rightOrder = (mit_sot_rightOrder +
                  sit_sot_rightOrder +
                  nit_sot_rightOrder)
    scan_out_list = [None] * len(rightOrder)
    for idx, pos in enumerate(rightOrder):
        if pos >= 0:
            scan_out_list[pos] = _scan_out_list[idx]
        else:
            # Not that pos is not a negative index. The sign of pos is used
            # as a flag to indicate if this output should be part of the
            # update rules or part of the standard outputs of scan.
            # If `pos` is positive than it corresponds to the standard
            # outputs of scan and it refers to output of index `pos`. If `pos`
            # is negative that it corresponds to update rules of scan and it
            # refers to update rule of index -1 - `pos`.
            update_map[sit_sot_shared[abs(pos) - 1]] = _scan_out_list[idx][-1]
    scan_out_list = [x for x in scan_out_list if x is not None]
    ################################################################## P2<
    return (scan_out_list, update_map)
