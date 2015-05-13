#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano.tensor as T

from . import NeuralLayer

class BatchNormalization(NeuralLayer):
    """
    Batch normalization.
    http://arxiv.org/pdf/1502.03167v3.pdf
    """
    def __init__(self, epsilon=1e-6, weights=None):
        super(BatchNormalization,self).__init__("norm")
        self.epsilon = epsilon

    def setup(self):
        self.gamma = self.create_weight(shape=(self.input_dim,), suffix="gamma")
        self.beta = self.create_bias(self.input_dim, suffix="beta")
        self.register_parameters(self.gamma, self.beta)

    def output(self, x):

        m = x.mean(axis=0)
        std = T.mean((x-m)**2 + self.epsilon, axis=0) ** 0.5
        x_normed = (x - m) / (std + self.epsilon)
        out = self.gamma * x_normed + self.beta
        return out