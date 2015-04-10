#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import NeuralLayer

class RevealDimension(NeuralLayer):
    """
    Operation for revealing dimension.
    After some dimension-unclear layers such as convolution, the dimension information will be lost.
    Use this layer to redefine the input dimension.
    """

    def __init__(self, dim):
        super(RevealDimension, self).__init__("reveal_dimension")
        self.dim = dim

    def setup(self):
        self.output_dim = self.dim

    def output(self, x):
        return x