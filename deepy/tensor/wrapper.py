#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano.tensor as T
from theano import tensor as theano_tensor

from deepy.core.tensor_conversion import neural_computation
from deepy.core.neural_var import NeuralVariable


class NeuralTensorNet(object):

    def __getattr__(self, func_name):
        @neural_computation
        def wrapper(*args, **kwargs):
            return getattr(theano_tensor.nnet, func_name)(*args, **kwargs)
        return wrapper

deepy_nnet = NeuralTensorNet()

class NeuralTensor(object):
    """
    A class for exporting Theano tensor operations to neural variables.

    """

    def constant(self, value, dtype="float32", dim=None):
        return NeuralVariable(T.constant(value, dtype=dtype), dim=dim)

    def __getattr__(self, func_name):
        global deepy_nnet
        @neural_computation
        def wrapper(*args, **kwargs):
            return getattr(theano_tensor, func_name)(*args, **kwargs)
        if func_name == 'nnet':
            return deepy_nnet
        else:
            return wrapper


deepy_tensor = NeuralTensor()
tensor = deepy_tensor

