#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np
import theano.tensor as T

from deepy.core.neural_var import NeuralVariable
from deepy.core.decorations import neural_computation
from deepy.utils import build_activation, FLOATX, XavierGlorotInitializer, OrthogonalInitializer, Scanner
from . import NeuralLayer

OUTPUT_TYPES = ["sequence", "one"]
INPUT_TYPES = ["sequence", "one"]



class RecurrentLayer(NeuralLayer):
    __metaclass__ = ABCMeta

    def __init__(self, name, state_names, hidden_size=100, input_type="sequence", output_type="sequence",
                 inner_init=None, outer_init=None,
                 gate_activation='sigmoid', activation='tanh',
                 steps=None, backward=False, mask=None,
                 additional_input_dims=None):
        super(RecurrentLayer, self).__init__(name)
        self.state_names = state_names
        self.main_state = state_names[0]
        self.hidden_size = hidden_size
        self._gate_activation = gate_activation
        self._activation = activation
        self.gate_activate = build_activation(self._gate_activation)
        self.activate = build_activation(self._activation)
        self._input_type = input_type
        self._output_type = output_type
        self.inner_init = inner_init if inner_init else OrthogonalInitializer()
        self.outer_init = outer_init if outer_init else XavierGlorotInitializer()
        self._steps = steps
        self._mask = mask.tensor if type(mask) == NeuralVariable else mask
        self._go_backwards = backward
        self.additional_input_dims = additional_input_dims if additional_input_dims else []

        if input_type not in INPUT_TYPES:
            raise Exception("Input type of {} is wrong: {}".format(name, input_type))
        if output_type not in OUTPUT_TYPES:
            raise Exception("Output type of {} is wrong: {}".format(name, output_type))

    @neural_computation
    def step(self, step_inputs):
        new_states = self.compute_new_state(step_inputs)

        # apply mask for each step if `output_type` is 'one'
        if step_inputs.get("mask"):
            mask = step_inputs["mask"].dimshuffle(0, 'x')
            for state_name in new_states:
                new_states[state_name] = new_states[state_name] * mask + step_inputs[state_name] * (1 - mask)

        return new_states

    @abstractmethod
    def compute_new_state(self, step_inputs):
        """
        :type step_inputs: dict
        :rtype: dict
        """

    @abstractmethod
    def merge_inputs(self, input_var, additional_inputs=None):
        """
        Merge inputs and return a map, which will be passed to core_step.
        :type input_var: T.var
        :param additional_inputs: list
        :rtype: dict
        """

    @abstractmethod
    def prepare(self):
        pass

    @neural_computation
    def compute_step(self, state, lstm_cell=None, input=None, additional_inputs=None):
        """
        Compute one step in the RNN.
        :return: one variable for RNN and GRU, multiple variables for LSTM
        """
        input_map = self.merge_inputs(input, additional_inputs=additional_inputs)
        input_map.update({"state": state, "lstm_cell": lstm_cell})
        output_map = self.compute_new_state(input_map)
        outputs = [output_map.pop("state")]
        outputs += output_map.values()
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    @neural_computation
    def get_initial_states(self, input_var, init_state=None):
        """
        :type input_var: T.var
        :rtype: dict
        """
        initial_states = {}
        for state in self.state_names:
            if state != "state" or not init_state:
                if self._input_type == 'sequence' and input_var.ndim == 2:
                    init_state = T.alloc(np.cast[FLOATX](0.), self.hidden_size)
                else:
                    init_state = T.alloc(np.cast[FLOATX](0.), input_var.shape[0], self.hidden_size)
            initial_states[state] = init_state
        return initial_states

    @neural_computation
    def get_step_inputs(self, input_var, states=None, mask=None, additional_inputs=None):
        """
        :type input_var: T.var
        :rtype: dict
        """
        step_inputs = {}
        if self._input_type == "sequence":
            if not additional_inputs:
                additional_inputs = []
            if mask:
                step_inputs['mask'] = mask.dimshuffle(1, 0)
            step_inputs.update(self.merge_inputs(input_var, additional_inputs=additional_inputs))
        else:
            # step_inputs["mask"] = mask.dimshuffle((1,0)) if mask else None
            if additional_inputs:
                step_inputs.update(self.merge_inputs(None, additional_inputs=additional_inputs))
        if states:
            for name in self.state_names:
                step_inputs[name] = states[name]

        return step_inputs

    def compute(self, input_var, mask=None, additional_inputs=None, steps=None, backward=False, init_states=None, return_all_states=False):
        if additional_inputs and not self.additional_input_dims:
            self.additional_input_dims = map(lambda var: var.dim(), additional_inputs)
        result_var = super(RecurrentLayer, self).compute(input_var,
                                                   mask=mask, additional_inputs=additional_inputs, steps=steps, backward=backward, init_states=init_states, return_all_states=return_all_states)
        if return_all_states:
            state_map = {}
            for k in result_var.tensor:
                state_map[k] = NeuralVariable(result_var.tensor[k], self.output_dim)
            return state_map
        else:
            return result_var

    def compute_tensor(self, input_var, mask=None, additional_inputs=None, steps=None, backward=False, init_states=None, return_all_states=False):
        # prepare parameters
        backward = backward if backward else self._go_backwards
        steps = steps if steps else self._steps
        mask = mask if mask else self._mask
        if mask and self._input_type == "one":
            raise Exception("Mask only works with sequence input")
        # get initial states
        init_state_map = self.get_initial_states(input_var)
        if init_states:
            for name, val in init_states.items():
                if name in init_state_map:
                    init_state_map[name] = val
        # get input sequence map
        if self._input_type == "sequence":
            # Move middle dimension to left-most position
            # (sequence, batch, value)
            if input_var.ndim == 3:
                input_var = input_var.dimshuffle((1,0,2))

            seq_map = self.get_step_inputs(input_var, mask=mask, additional_inputs=additional_inputs)
        else:
            init_state_map[self.main_state] = input_var
            seq_map = self.get_step_inputs(None, mask=mask, additional_inputs=additional_inputs)
        # scan
        retval_map, _ = Scanner(
            self.step,
            sequences=seq_map,
            outputs_info=init_state_map,
            n_steps=steps,
            go_backwards=backward
        ).compute()
        # return main states
        main_states = retval_map[self.main_state]
        if self._output_type == "one":
            if return_all_states:
                return_map = {}
                for name, val in retval_map.items():
                    return_map[name] = val[-1]
                return return_map
            else:
                return main_states[-1]
        elif self._output_type == "sequence":
            if return_all_states:
                return_map = {}
                for name, val in retval_map.items():
                    return_map[name] = val.dimshuffle((1,0,2))
                return return_map
            else:
                main_states = main_states.dimshuffle((1,0,2)) # ~ batch, time, size
                # if mask: # ~ batch, time
                #     main_states *= mask.dimshuffle((0, 1, 'x'))
                return main_states


class RNN(RecurrentLayer):

    def  __init__(self, hidden_size, **kwargs):
        kwargs["hidden_size"] = hidden_size
        super(RNN, self).__init__("RNN", ["state"], **kwargs)

    @neural_computation
    def compute_new_state(self, step_inputs):
        xh_t, h_tm1 = map(step_inputs.get, ["xh_t", "state"])
        if not xh_t:
            xh_t = 0

        h_t = self.activate(xh_t + T.dot(h_tm1, self.W_h) + self.b_h)

        return {"state": h_t}

    @neural_computation
    def merge_inputs(self, input_var, additional_inputs=None):
        if not additional_inputs:
            additional_inputs = []
        all_inputs = ([input_var] if input_var else []) + additional_inputs
        h_inputs = []
        for x, weights in zip(all_inputs, self.input_weights):
            wi, = weights
            h_inputs.append(T.dot(x, wi))
        merged_inputs = {
            "xh_t": sum(h_inputs)
        }
        return merged_inputs

    def prepare(self):
        self.output_dim = self.hidden_size

        self.W_h = self.create_weight(self.hidden_size, self.hidden_size, "h", initializer=self.outer_init)
        self.b_h = self.create_bias(self.hidden_size, "h")

        self.register_parameters(self.W_h, self.b_h)

        self.input_weights = []
        if self._input_type == "sequence":
            normal_input_dims = [self.input_dim]
        else:
            normal_input_dims = []

        all_input_dims = normal_input_dims + self.additional_input_dims
        for i, input_dim in enumerate(all_input_dims):
            wi = self.create_weight(input_dim, self.hidden_size, "wi_{}".format(i+1), initializer=self.outer_init)
            weights = [wi]
            self.input_weights.append(weights)
            self.register_parameters(*weights)