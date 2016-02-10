#!/usr/bin/env python
# -*- coding: utf-8 -*-

from layer import NeuralLayer
from deepy.utils.decorations import neural_computation


class NeuralVariable(NeuralLayer):
    """
    Create a constant layer with tensors.
    """

    def __init__(self, tensor, test_tensor=None, dim=0):
        """
        Create a tensor layer.
        """
        super(NeuralVariable, self).__init__("const")
        self.output_dim = dim
        self.tensor = tensor
        self.test_tensor = tensor if not test_tensor else test_tensor
        self.initialize(0)

    def __getitem__(self, index):
        @neural_computation
        def getitem_wrapper(t, index):
            if type(index) == list:
                index = tuple(index)
            return t.__getitem__(index)
        return getitem_wrapper(self, index)

    def apply(self, func, dim=None):
        """
        Apply a function to tensors.
        """
        output_dim = dim if dim else self.output_dim
        return NeuralVariable(func(self.tensor), func(self.test_tensor), output_dim)

    def compute_tensor(self, x):
        return self.tensor

    def compute_test_tesnor(self, x):
        return self.test_tensor

    def set_test_value(self, value):
        self.tensor.tag.test_value = value

    def dim(self):
        return self.output_dim

    def shape(self, dim_index):
        return NeuralVariable(self.tensor.shape[dim_index], self.test_tensor.shape[dim_index])
