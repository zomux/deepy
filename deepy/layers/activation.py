#!/usr/bin/env python
# -*- coding: utf-8 -*-

from layer import NeuralLayer

class Activation(NeuralLayer):
    """
    Activation layer.
    """

    def __init__(self, activation_type):
        from deepy.tensor.activations import get_activation
        super(Activation, self).__init__(activation_type)
        self._activation = get_activation(activation_type)

    def compute_tensor(self, x):
        self.activation = self._activation(x)
        return self.activation