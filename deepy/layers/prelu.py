#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import NeuralLayer

class PRelu(NeuralLayer):
    """
    Probabilistic ReLU.
     - http://arxiv.org/pdf/1502.01852v1.pdf
    """
    def __init__(self):
        super(PRelu, self).__init__("prelu")

    def setup(self):
        self.alphas = self.create_bias(self.output_dim, "alphas")
        self.register_parameters(self.alphas)

    def output(self, x):
        positive_vector =  x * (x >= 0)
        negative_vector = self.alphas * (x * (x < 0))
        return positive_vector + negative_vector