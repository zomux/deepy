#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy.layers import NeuralLayer
from deepy.utils import build_activation
import theano.tensor as T

class HighwayLayer(NeuralLayer):
    """
    Highway network layer.
    See http://arxiv.org/abs/1505.00387.
    """

    def __init__(self, activation='relu', init=None, gate_bias=-5):
        super(HighwayLayer, self).__init__("highway")
        self.activation = activation
        self.init = init
        self.gate_bias = gate_bias

    def prepare(self):
        self.output_dim = self.input_dim
        self._act = build_activation(self.activation)
        self.W_h = self.create_weight(self.input_dim, self.input_dim, "h", initializer=self.init)
        self.W_t = self.create_weight(self.input_dim, self.input_dim, "t", initializer=self.init)
        self.B_h = self.create_bias(self.input_dim, "h")
        self.B_t = self.create_bias(self.input_dim, "t", value=self.gate_bias)

        self.register_parameters(self.W_h, self.B_h, self.W_t, self.B_t)

    def compute_tensor(self, x):
        t = self._act(T.dot(x, self.W_t) + self.B_t)
        h = self._act(T.dot(x, self.W_h) + self.B_h)
        return h * t + x * (1 - t)

