#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
from map_dict import MapDict

class Scanner(object):
    """
    Call `theano.scan` with dictionary parameters.
    """

    def __init__(self, func, sequences=None, outputs_info=None, non_sequences=None, neural_computation=False, **kwargs):
        if ((sequences and type(sequences) != dict) or
            (outputs_info and type(outputs_info) != dict) or
            (non_sequences and type(non_sequences) != dict)):
            raise Exception("The parameter `sequences`, `outputs_info`, `non_sequences` must be dict.")
        self._func = func

        self._sequence_keys = sequences.keys() if sequences else []
        self._sequence_values = sequences.values() if sequences else []
        self._output_keys = outputs_info.keys() if outputs_info else []
        self._output_values = outputs_info.values() if outputs_info else []
        self._non_sequence_keys = non_sequences.keys() if non_sequences else []
        self._non_sequence_values = non_sequences.values() if non_sequences else []
        self._kwargs = kwargs
        self._neural_computation = neural_computation
        self._input_dim_list = []
        self._output_dim_map = {}

    def _func_wrapper(self, *vars):
        from deepy.core.tensor_conversion import convert_to_theano_var, convert_to_neural_var
        all_values = self._sequence_values + self._output_values + self._non_sequence_values
        all_keys = self._sequence_keys + self._output_keys + self._non_sequence_keys
        valid_keys = [all_keys[i] for i in range(len(all_keys)) if all_values[i] is not None]
        none_keys = [all_keys[i] for i in range(len(all_keys)) if all_values[i] is None]
        if self._neural_computation:
            for var, last_dim in zip(vars, self._input_dim_list):
                var.tag.last_dim = last_dim

        dict_param = MapDict(zip(valid_keys, vars))
        dict_param.update(MapDict(zip(none_keys, [None for _ in range(len(none_keys))])))
        if self._neural_computation:
            dict_param = convert_to_neural_var(dict_param)
        retval = self._func(dict_param)
        if type(retval) == tuple:
            dict_retval, updates = retval
        else:
            dict_retval, updates = retval, None
        if self._neural_computation:
            if isinstance(dict_retval, dict):
                for k, var in dict_retval.items():
                    self._output_dim_map[k] = var.dim()
            updates, _, _ = convert_to_theano_var(updates)
            dict_retval, _, _ = convert_to_theano_var(dict_retval)
        if type(dict_retval) == MapDict:
            dict_retval = dict(dict_retval.items())
        if type(dict_retval) != dict:
            raise Exception("The return value of scanner function must be a dict")
        final_retval = [dict_retval[k] for k in self._output_keys]
        if len(final_retval) == 1:
            final_retval = final_retval[0]
        if updates:
            return final_retval, updates
        else:
            return final_retval

    def compute(self):
        from deepy.core.tensor_conversion import convert_to_theano_var, convert_to_neural_var
        if self._neural_computation:
            self._input_dim_list = []
            for tensor in sum([self._sequence_values, self._output_values, self._non_sequence_values], []):
                last_dim = tensor.tag.last_dim if tensor and hasattr(tensor.tag, 'last_dim') else None
                self._input_dim_list.append(last_dim)
        results, updates = theano.scan(self._func_wrapper,
                            sequences=filter(lambda t: t is not None, self._sequence_values),
                            outputs_info=self._output_values,
                            non_sequences=filter(lambda t: t is not None, self._non_sequence_values),
                            **self._kwargs)
        if type(results) != list:
            results = [results]
        result_dict = MapDict(zip(self._output_keys, results))
        if self._neural_computation:
            result_dict = convert_to_neural_var(result_dict)
            for k in result_dict:
                if k in self._output_dim_map:
                    result_dict[k].output_dim = self._output_dim_map[k]
        return result_dict, updates