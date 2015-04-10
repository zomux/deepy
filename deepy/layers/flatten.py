#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import NeuralLayer
import theano.tensor as T

class Flatten(NeuralLayer):
    """
    Flatten layer.
    """

    def __init__(self):
        super(Flatten, self).__init__("flatten")
    
    def output(self, x):
        return T.flatten(x, 2)