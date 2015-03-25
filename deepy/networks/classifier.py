#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from deepy.networks.basic_nn import NeuralNetwork
import theano.tensor as T

class NeuralClassifier(NeuralNetwork):
    '''A classifier attempts to match a 1-hot target output.'''

    def __init__(self, config):
        super(NeuralClassifier, self).__init__(config)

    def setup_vars(self):
        super(NeuralClassifier, self).setup_vars()

        # for a classifier, k specifies the correct labels for a given input.
        self.vars.k = T.ivector('k')
        self.inputs.append(self.vars.k)
        self.target_inputs.append(self.vars.k)

    @staticmethod
    def _cost_func(self):
            return -T.mean(T.log(self.vars.y)[T.arange(self.vars.k.shape[0]), self.vars.k])

    @staticmethod
    def _error_func(self):
        return 100 * T.mean(T.neq(T.argmax(self.vars.y, axis=1), self.vars.k))

    @property
    def cost(self):
        return self._cost_func(self)

    @cost.setter
    def cost(self, value):
        self._cost_func = value

    @property
    def errors(self):
        '''Compute the percent correct classifications.'''
        return self._error_func(self)

    @errors.setter
    def errors(self, value):
      self._error_func = value

    @property
    def monitors(self):
        yield 'err', self.errors
        for i, h in enumerate(self.hiddens):
            yield 'h{}<0.1'.format(i+1), 100 * (abs(h) < 0.1).mean()
            yield 'h{}<0.9'.format(i+1), 100 * (abs(h) < 0.9).mean()
        for name, exp in self.special_monitors:
            yield name, exp

    def classify(self, x):
        return self.predict(x).argmax(axis=1)
