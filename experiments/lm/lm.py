#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy.networks import NeuralNetwork
from deepy.utils import onehot, EPSILON
import theano.tensor as T

class NeuralLM(NeuralNetwork):
    """
    LM Network.
    """

    def setup_variables(self):
        super(NeuralLM, self).setup_variables()

        self.k = T.imatrix('k')
        self.target_variables.append(self.k)

    def _cost_func(self, y):
        y = T.clip(y, EPSILON, 1.0 - EPSILON)
        y2 = y.reshape((-1, y.shape[-1]))
        k2 = self.k.reshape((-1,))
        return -T.mean(T.log(y2)[T.arange(k2.shape[0]), k2])

    def _error_func(self, y):
        y2 = y.reshape((-1, y.shape[-1]))
        k2 = self.k.reshape((-1,))
        return 100 * T.mean(T.neq(T.argmax(y2[:k2.shape[0]], axis=1), k2))

    def _perplexity_func(self, y):
        return 2**self._cost_func(y)

    @property
    def cost(self):
        return self._cost_func(self.output)

    @property
    def test_cost(self):
        return self._cost_func(self.test_output)

    def predict(self, x):
        return self.compute(x).argmax(axis=1)

    def sample(self, input, steps):
        """
        Sample outputs from LM.
        """
        inputs = [[onehot(self.input_dim, x) for x in input]]
        for _ in range(steps):
            target = self.compute(inputs)[0,-1].argmax()
            input.append(target)
            inputs[0].append(onehot(self.input_dim, target))
        return input


    def prepare_training(self):
        self.training_monitors.append(("err", self._error_func(self.output)))
        self.testing_monitors.append(("err", self._error_func(self.test_output)))
        self.training_monitors.append(("PPL", self._perplexity_func(self.output)))
        self.testing_monitors.append(("PPL", self._perplexity_func(self.test_output)))
        super(NeuralLM, self).prepare_training()