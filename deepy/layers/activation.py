#!/usr/bin/env python
# -*- coding: utf-8 -*-

from layer import NeuralLayer
from deepy.utils import get_activation

class Activation(NeuralLayer):
    """
    Activation layer.
    """

    def __init__(self, activation_type):
        super(Activation, self).__init__(activation_type)
        self._activation = get_activation(activation_type)

    def compute_tensor(self, x):
        return self._activation(x)