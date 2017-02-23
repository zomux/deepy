#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy.utils import Scanner, MapDict
from dummy_loop_utils import get_dummy_args, finish_scan
import theano

class LoopVars(MapDict):
    """
    Variables inside the loop.
    """

class Loop(object):

    def __init__(self, sequences=None, outputs=None, non_sequences=None, block=None, **kwargs):
        """
        A loop function to support "with" grammar.
        """
        self._sequences = sequences if sequences else {}
        self._outputs = outputs if outputs else {}
        self._non_sequences = non_sequences if non_sequences else {}
        if not isinstance(self._sequences, dict) or not isinstance(self._outputs, dict) != dict or not isinstance(self._non_sequences, dict):
            raise Exception("Arguments of Loop shall be dicts.")
        self._block = block
        self._kwargs = kwargs
        self._loop_vars = None
        self._dummy_nodes = None
        self._scan_local_vars = None
        self._ordered_out_keys = []
        self._scan_outputs = None
        self._scan_updates = None

    def _build_loop_vars(self):
        """
        Create inner loop variables.
        """
        from theano.tensor.var import TensorVariable
        from deepy.core.neural_var import NeuralVariable
        if not self._loop_vars:
            self._ordered_out_keys = self._outputs.keys()
            seq_keys = self._sequences.keys()
            filled_out_keys = [k for k in self._ordered_out_keys if self._outputs[k]]
            nonseq_keys = self._non_sequences.keys()
            dummy_tensors, self._scan_local_vars = get_dummy_args(
                sequences=[self._sequences[k].tensor for k in seq_keys],
                outputs_info=[self._outputs[k].tensor if self._outputs[k] else None for k in self._ordered_out_keys],
                non_sequences=[self._non_sequences[k].tensor for k in nonseq_keys],
                **self._kwargs
            )
            dummy_map = dict(zip(seq_keys + filled_out_keys + nonseq_keys, dummy_tensors))
            arg_map = self._sequences.copy()
            arg_map.update(self._outputs)
            arg_map.update(self._non_sequences)
            self._loop_vars = LoopVars()
            for k, dummy_tensor in dummy_map.items():
                dummy_var = NeuralVariable(dummy_tensor, dim=arg_map[k].dim())
                self._loop_vars[k] = dummy_var
            # self._dummy_nodes = dict(self._loop_vars.items()[:])


    def __enter__(self):
        self._build_loop_vars()
        return self._loop_vars

    def __exit__(self, exc_type, exc_val, exc_tb):
        from neural_var import NeuralVariable
        output_tensors = []
        for k in self._ordered_out_keys:
            if k not in self._loop_vars:
                raise Exception("{} can not be found in loop vars.".format(k))
            tensor_k = self._loop_vars[k].tensor if isinstance(self._loop_vars[k], NeuralVariable) else self._loop_vars[k]
            output_tensors.append(tensor_k)

        result_tensors, updates = finish_scan(output_tensors, self._scan_local_vars)
        if self._block and updates:
            if type(updates) == dict:
                updates = updates.items()
            self._block.register_updates(*updates)

        outputs = MapDict()
        for k, tensor in zip(self._ordered_out_keys, result_tensors):
            out_var = NeuralVariable(tensor)
            if self._outputs[k] is not None:
                out_var.output_dim = self._outputs[k].dim()
            outputs[k] = out_var
        self._scan_outputs = outputs
        self._scan_updates = updates.items()

    def _scan_step(self, vars):
        """
        Internal scan with dummy input variables.
        """
        from neural_var import NeuralVariable
        if not self._loop_vars:
            raise Exception("The loop is not initialized. To initialize the loop, use `with loop as vars`")
        replace_map = {}
        for k, var in vars.items():
            if var is not None:
                replace_map[self._dummy_nodes[k].tensor] = var.tensor
        outputs = {}
        for k in self._outputs:
            if k not in self._loop_vars:
                raise Exception("{} can not be found in loop vars.".format(k))
            output_node = theano.clone(self._loop_vars[k].tensor, replace_map)
            outputs[k] = NeuralVariable(output_node, self._loop_vars[k].dim())
        return outputs

    @property
    def outputs(self):
        """
        Get the output of the loop.
        :rtype: MapDict
        """
        if not self._scan_outputs:
            raise Exception("The loop is not executed.")
        else:
            return self._scan_outputs
        
    @property
    def updates(self):
        """
        Get the updates of the loop.
        :rtype: MapDict
        """
        if not self._scan_outputs:
            raise Exception("The loop is not executed.")
        else:
            return self._scan_updates

    def get_outputs(self, *args):
        """
        Get the outputs of the loop.
        Return specific variables by passing the keys to the arguments.
        :rtype: MapDict
        """
        if args:
            output_vars = map(self._scan_outputs.get, args)
            if len(output_vars) == 1:
                return output_vars[0]
            else:
                return output_vars
        else:
            return self._scan_outputs


