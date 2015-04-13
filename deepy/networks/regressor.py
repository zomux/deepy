#!/usr/bin/env python
# -*- coding: utf-8 -*-

from theano import tensor as T
from . import NeuralNetwork
from deepy.util import dim_to_var


class NeuralRegressor(NeuralNetwork):
    """
    Regression network.
    """
    def __init__(self, input_dim, config=None, input_tensor=None, target_tensor=2):
        self.target_tensor = dim_to_var(target_tensor, "k") if type(target_tensor) == int else target_tensor
        super(NeuralRegressor, self).__init__(input_dim, config=config, input_tensor=input_tensor)

    def setup_variables(self):
        super(NeuralRegressor, self).setup_variables()
        self.k = self.target_tensor
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