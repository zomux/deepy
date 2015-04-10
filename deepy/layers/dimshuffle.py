#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import NeuralLayer

class DimShuffle(NeuralLayer):
    """
    DimShuffle layer.
    """

    def __init__(self, pattern):
        super(DimShuffle, self).__init__("dimshuffle")
        self.pattern = pattern

    def output(self, x):
        return x.dimshuffle(*self.pattern)