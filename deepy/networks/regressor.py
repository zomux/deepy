#!/usr/bin/env python
# -*- coding: utf-8 -*-

from theano import tensor as T
from network import NeuralNetwork


class NeuralRegressor(NeuralNetwork):
    """
    Regression network.
    """

    def setup_vars(self):
        super(NeuralRegressor, self).setup_variables()
        self.k = T.matrix('k')
        self.target_variables.append(self.k)

    def _cost_func(self, y):
        err = y - self.k
        return T.mean((err * err).sum(axis=1))

    @property
    def cost(self):
        return self._cost_func(self.output)

    @property
    def test_cost(self):
        return self._cost_func(self.test_output)