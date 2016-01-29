#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import NeuralLayer

class DimShuffle(NeuralLayer):
    """
    DimShuffle layer.
    """

    def __init__(self, *pattern):
        super(DimShuffle, self).__init__("dimshuffle")
        if len(pattern) == 1 and type(pattern[0]) == list:
            self.pattern = pattern[0]
        else:
            self.pattern = pattern

    def compute_tensor(self, x):
        return x.dimshuffle(*self.pattern)