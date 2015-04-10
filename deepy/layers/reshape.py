#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import NeuralLayer

class Reshape(NeuralLayer):
    """
    Reshape layer.
    """

    def __init__(self, pattern):
        super(Reshape, self).__init__("dimshuffle")
        self.pattern = pattern

    def output(self, x):
        return x.reshape(self.pattern)