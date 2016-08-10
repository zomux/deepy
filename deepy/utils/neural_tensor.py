#!/usr/bin/env python
# -*- coding: utf-8 -*-

from theano import tensor as theano_tensor
from decorations import neural_computation

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
tensor = NT = deepy_tensor

