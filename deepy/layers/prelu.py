#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import NeuralLayer
from conv import Convolution

class PRelu(NeuralLayer):
    """
    Probabilistic ReLU.
     - http://arxiv.org/pdf/1502.01852v1.pdf
    """
    def __init__(self, input_tensor=2):
        super(PRelu, self).__init__("prelu")
        self.input_tensor = input_tensor

    def setup(self):
        self.alphas = self.create_bias(self.output_dim, "alphas")
        self.register_parameters(self.alphas)
        if self.input_tensor == 3:
            self.alphas = self.alphas.dimshuffle('x', 0, 'x')
        elif self.input_tensor == 4:
            self.alphas = self.alphas.dimshuffle('x', 0, 'x', 'x')

    def output(self, x):
        positive_vector =  x * (x >= 0)
        negative_vector = self.alphas * (x * (x < 0))
        return positive_vector + negative_vector