#!/usr/bin/env python
# -*- coding: utf-8 -*-

from layer import NeuralLayer
import theano.tensor as T

class Softmax(NeuralLayer):

    def __init__(self):
        super(Softmax, self).__init__("softmax")

    def output(self, x):
        return T.nnet.softmax(x)