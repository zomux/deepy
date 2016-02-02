#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
from decorations import neural_computation

class Scanner(object):
    """
    Call `theano.scan` with dictionary parameters.
    """

    def __init__(self, func, sequences=None, outputs_info=None, non_sequences=None, **kwargs):
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


    def _func_wrapper(self, *vars):
        all_values = self._sequence_values + self._output_values + self._non_sequence_values
        all_keys = self._sequence_keys + self._output_keys + self._non_sequence_keys
        valid_keys = [all_keys[i] for i in range(len(all_keys)) if all_values[i] is not None]
        none_keys = [all_keys[i] for i in range(len(all_keys)) if all_values[i] is None]

        dict_param = dict(zip(valid_keys, vars))
        dict_param.update(dict(zip(none_keys, [None for _ in range(len(none_keys))])))
        retval = self._func(dict_param)
        if type(retval) == tuple:
            dict_retval, updates = retval
        else:
            dict_retval, updates = retval, None
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
        results, updates = theano.scan(self._func_wrapper,
                            sequences=filter(lambda t: t is not None, self._sequence_values),
                            outputs_info=self._output_values,
                            non_sequences=filter(lambda t: t is not None, self._non_sequence_values),
                            **self._kwargs)
        if type(results) != list:
            results = [results]
        return dict(zip(self._output_keys, results)), updates
