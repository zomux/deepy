#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano.tensor as TT
from theano.ifelse import ifelse as theano_ifelse

from deepy.core.tensor_conversion import neural_computation
from deepy.layers.concat import Concatenate
from wrapper import deepy_tensor


def concat(vars, axis=-1):
    """
    A shortcut for concatenation.
    """
    return concatenate(vars, axis)

@neural_computation
def reverse(tensor, axis=-1):
    ndim = tensor.ndim
    selectors = [slice(None)] * ndim
    selectors[axis] = slice(None, None, -1)
    ret = tensor[selectors]
    if hasattr(tensor.tag, "last_dim"):
        ret.tag.last_dim = tensor.tag.last_dim
    return ret

@neural_computation
def activate(var, method):
    """
    An activation function.
    :param var: input var
    :param method: type of activation, such as `relu`,`tanh`,`sigmoid`
    """
    from activations import get_activation
    return get_activation(method)(var)

def concatenate(vars, axis=-1):
    """
    A utility function of concatenate.
    """
    from deepy.core.neural_var import NeuralVariable
    if isinstance(vars[0], NeuralVariable):
        concat_var = Concatenate(axis=axis).compute(*vars)
        if axis == -1 or axis == vars[0].tensor.ndim - 1:
            concat_var.output_dim = sum([x.output_dim for x in vars], 0)
    else:
        concat_var = TT.concatenate(vars, axis)
    return concat_var

@neural_computation
def ifelse(condition, then_branch, else_branch):
    return theano_ifelse(condition, then_branch, else_branch)


def apply(func, *args, **kwargs):
    from deepy.core.neural_var import NeuralVariable
    dim = kwargs['dim'] if 'dim' in kwargs else args[0].dim()
    return NeuralVariable(func(*[x.tensor for x in args]), dim)

def repeat(*args, **kwargs):
    return deepy_tensor.repeat(*args, **kwargs)

def vars(*tensor_types):
    """
    Create multiple variables without specifying last dimension and shape.
    :rtype: list of deepy.core.neural_var.NeuralVariable
    """
    return map(var, tensor_types)


def var(tensor_type, last_dim=0, test_shape=None):
    """
    Wrap a Theano tensor into the variable for defining neural network.
    :param last_dim: last dimension of tensor, 0 indicates that the last dimension is flexible
    :rtype: deepy.core.neural_var.NeuralVariable
    """
    # Create tensor
    from deepy.core.neural_var import NeuralVariable
    from deepy.core.env import env
    from theano.tensor.var import TensorVariable
    if isinstance(tensor_type, NeuralVariable):
        var = tensor_type
        if last_dim != 0:
            var.output_dim = last_dim
    elif isinstance(tensor_type, TensorVariable):
        var = NeuralVariable(tensor_type, dim=last_dim)
    elif isinstance(tensor_type, str):
        theano_tensor = getattr(TT, tensor_type)()
        var = NeuralVariable(theano_tensor, dim=last_dim)
    else:
        raise Exception("tensor_type shall be a string or a NeuralVariable")
    # Set test value
    if test_shape:
        if type(test_shape) != list and type(test_shape) != tuple:
            # May be it's a value
            var.set_test_value(test_shape)
        else:
            test_val = env.numpy_rand.rand(*test_shape)
            if len(test_shape) > 0:
                test_val = test_val.astype(var.tensor.dtype)
            elif var.tensor.dtype.startswith("int"):
                test_val = 1
            var.set_test_value(test_val)
    else:
        # Create a general test_shape
        dims = [(d + 1) * 3 for d in range(var.tensor.ndim)]
        if var.dim() != 0:
            dims[-1] = var.dim()
        test_val = env.numpy_rand.rand(*dims)
        if len(dims) > 0:
            test_val = test_val.astype(var.tensor.dtype)
        elif var.tensor.dtype.startswith("int"):
            test_val = 1
        var.set_test_value(test_val)
    return var


def is_neural_var(var):
    from deepy.core.neural_var import NeuralVariable
    return isinstance(var, NeuralVariable)

def is_theano_var(var):
    from theano.tensor.var import TensorVariable
    return isinstance(var, TensorVariable)