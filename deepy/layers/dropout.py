#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import NeuralLayer
from deepy.utils import global_theano_rand, FLOATX

class Dropout(NeuralLayer):

    def __init__(self, p):
        super(Dropout, self).__init__("dropout")
        self.p = p

    def output(self, x):
        if self.p > 0:
            x *= global_theano_rand.binomial(x.shape, p=1-self.p, dtype=FLOATX)
        return x

    def test_output(self, x):
        if self.p > 0:
            x *= (1.0 - self.p)
        return x
