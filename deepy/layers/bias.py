#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import NeuralLayer

class Bias(NeuralLayer):
    """
    Bias layer.
    """

    def __init__(self):
        super(Bias, self).__init__("bias")


    def prepare(self):
        self.output_dim = self.input_dim
        self.B = self.create_bias(self.output_dim, "bias")
        self.register_parameters(self.B)

    def compute_tensor(self, x):
        return x + self.B