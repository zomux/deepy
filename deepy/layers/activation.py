#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import NeuralLayer
from deepy.utils import build_activation

class Activation(NeuralLayer):
    """
    Activation layer.
    """

    def __init__(self, activation_type):
        super(Activation, self).__init__(activation_type)
        self._activation = build_activation(activation_type)

    def output(self, x):
        return self._activation(x)