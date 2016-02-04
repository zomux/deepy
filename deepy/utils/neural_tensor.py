#!/usr/bin/env python
# -*- coding: utf-8 -*-

from theano import tensor
from decorations import neural_computation

class NeuralTensor():
    """
    A class for exporting Theano tensor operations to neural variables.
    """

    def __getattr__(self, func_name):
        @neural_computation
        def wrapper(*args, **kwargs):
            return getattr(tensor, func_name)(*args, **kwargs)
        return wrapper


neural_tensor = NeuralTensor()
NT = neural_tensor