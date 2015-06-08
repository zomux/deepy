#!/usr/bin/env python
# -*- coding: utf-8 -*-

from layer import NeuralLayer
import theano
import theano.tensor as T

class Softmax3D(NeuralLayer):

    def __init__(self):
        super(Softmax3D, self).__init__("softmax")

    def output(self, x):
        shape = x.shape
        x = x.reshape((-1, shape[-1]))
        softmax_tensor = T.nnet.softmax(x)

        return softmax_tensor.reshape(shape)