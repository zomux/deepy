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

neural_tensor_net = NeuralTensorNet()

class NeuralTensor(object):
    """
    A class for exporting Theano tensor operations to neural variables.
    """

    def __getattr__(self, func_name):
        global neural_tensor_net
        @neural_computation
        def wrapper(*args, **kwargs):
            return getattr(theano_tensor, func_name)(*args, **kwargs)
        if func_name == 'nnet':
            return neural_tensor_net
        else:
            return wrapper


neural_tensor = NeuralTensor()
NT = neural_tensor
tensor = neural_tensor