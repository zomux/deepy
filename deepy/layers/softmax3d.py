#!/usr/bin/env python
# -*- coding: utf-8 -*-

from layer import NeuralLayer
import theano
import theano.tensor as T

class Softmax3D(NeuralLayer):

    def __init__(self):
        super(Softmax3D, self).__init__("softmax")

    def output(self, x):
        x = x.dimshuffle(1, 0, 2)
        softmax_tensor3, _ = theano.scan(lambda matrix: T.nnet.softmax(matrix), sequences=[x])
        softmax_tensor3.name = "softmax_loop"
        return softmax_tensor3.dimshuffle(0, 1, 2)