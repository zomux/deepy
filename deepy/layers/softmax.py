#!/usr/bin/env python
# -*- coding: utf-8 -*-

from layer import NeuralLayer
import theano.tensor as T

class Softmax(NeuralLayer):

    def output(self, x):
        return T.nnet.softmax(x)