#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano.tensor as T

from . import NeuralLayer

class BatchNormalization(NeuralLayer):
    """
    http://arxiv.org/pdf/1502.03167v3.pdf
    """
    def __init__(self, epsilon=1e-6, mode="featurewise", weights=None):
        super(BatchNormalization,self).__init__()
        self.epsilon = epsilon
        self.mode = mode

    def setup(self):
        self.gamma = self.create_weight(shape=(self.input_dim,), suffix="gamma")
        self.beta = self.create_bias(self.input_dim, suffix="beta")
        self.register_parameters(self.gamma, self.beta)

    def output(self, x):

        if self.mode == "featurewise":
            m = x.mean(axis=0)
            # Prevent NaNs
            std = T.mean((x-m)**2 + self.epsilon, axis=0) ** 0.5
            x_normed = (x - m) / (std + self.epsilon)

        else:
            m = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True)
            x_normed = (x - m) / (std + self.epsilon)

        out = self.gamma * x_normed + self.beta
        return out