#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano

from . import NeuralLayer
from deepy.utils import global_theano_rand, FLOATX

class Dropout(NeuralLayer):

    def __init__(self, p):
        super(Dropout, self).__init__("dropout")
        self.p = p

    def compute_tensor(self, x):
        if self.p > 0:
            # deal with the problem of test_value
            backup_test_value_setting = theano.config.compute_test_value
            theano.config.compute_test_value = 'ignore'
            binomial_mask = global_theano_rand.binomial(x.shape, p=1-self.p, dtype=FLOATX)
            theano.config.compute_test_value = backup_test_value_setting
            # apply dropout
            x *= binomial_mask
        return x

    def compute_test_tesnor(self, x):
        if self.p > 0:
            x *= (1.0 - self.p)
        return x
