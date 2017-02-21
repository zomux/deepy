#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
from layer import NeuralLayer

class Dropout(NeuralLayer):

    def __init__(self, p):
        super(Dropout, self).__init__("dropout")
        self.p = p

    def compute_tensor(self, x):
        from deepy.core import runtime, env
        if self.p > 0:
            # deal with the problem of test_value
            backup_test_value_setting = theano.config.compute_test_value
            theano.config.compute_test_value = 'ignore'
            binomial_mask = env.theano_rand.binomial(x.shape, p=1-self.p, dtype=env.FLOATX)
            theano.config.compute_test_value = backup_test_value_setting
            # apply dropout
            x = runtime.iftrain(x * binomial_mask, x * (1.0 - self.p))
        return x
