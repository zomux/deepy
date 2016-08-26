#!/usr/bin/env python
# -*- coding: utf-8 -*-

from layer import NeuralLayer
from deepy.utils.decorations import neural_computation, convert_to_theano_var


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
        ret = getitem_wrapper(self, index)
        if (hasattr(ret.tensor, 'tag') and hasattr(ret.tensor.tag, 'test_value')
            and ret.tensor.tag.test_value is not None and len(ret.tensor.tag.test_value.shape) > 0):
            ret.output_dim = ret.tensor.tag.test_value.shape[-1]
        else:
            ret.output_dim = self.dim()
        return ret

    def __call__(self, *args, **kwargs):
        normal_args, test_args, tensor_found_in_args, neural_found_in_args = convert_to_theano_var(args)
        normal_kwargs, test_kwargs, tensor_found_in_kwargs, neural_found_in_kwargs = convert_to_theano_var(kwargs)

        tensor_found = tensor_found_in_args or tensor_found_in_kwargs

        if tensor_found:
            raise Exception("Theano tensor variables can not be used together with neural variables.")

        return NeuralVariable(self.tensor(*normal_args, **normal_kwargs), self.test_tensor(*test_args, **test_kwargs), dim=self.dim())

    def __getattr__(self, name):
        return NeuralVariable(getattr(self.tensor, name), getattr(self.test_tensor, name), dim=self.dim())

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

    # def shape(self, dim_index):
    #     return NeuralVariable(self.tensor.shape[dim_index], self.test_tensor.shape[dim_index])

    def _other_tensor(self, other):
        return  other.tensor if isinstance(other, NeuralVariable) else other

    def _other_test_tensor(self, other):
        return other.test_tensor if isinstance(other, NeuralVariable) else other

    def __add__(self, other):

        return NeuralVariable(self.tensor + self._other_tensor(other), self.test_tensor + self._other_test_tensor(other), dim=self.dim())

    def __sub__(self, other):
        return NeuralVariable(self.tensor - self._other_tensor(other), self.test_tensor - self._other_test_tensor(other), dim=self.dim())

    def __mul__(self, other):
        return NeuralVariable(self.tensor * self._other_tensor(other), self.test_tensor * self._other_test_tensor(other), dim=self.dim())

    def __div__(self, other):
        return NeuralVariable(self.tensor / self._other_tensor(other), self.test_tensor / self._other_test_tensor(other), dim=self.dim())

    @property
    def test_value(self):
        if hasattr(self.tensor.tag, 'test_value'):
            return self.tensor.tag.test_value
        else:
            return None

    @property
    def tv(self):
        return self.test_value

    @property
    def ts(self):
        if self.test_value is not None:
            return self.test_value.shape
        else:
            return None