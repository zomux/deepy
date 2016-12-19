#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano.tensor as TT
from theano.ifelse import ifelse as theano_ifelse

from deepy.layers.concat import Concatenate
from deepy.core.decorations import neural_computation
from wrapper import deepy_tensor

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

def var(self, tensor_type, last_dim=0, test_shape=None):
    """
            Wrap a Theano tensor into the variable for defining neural network.
            :param last_dim: last dimension of tensor, 0 indicates that the last dimension is flexible
            :rtype: TensorVar
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
    # Create test value
    if test_shape:
        if type(test_shape) != list and type(test_shape) != tuple:
            var.set_test_value(test_shape)
        else:
            var.set_test_value(env.numpy_rand.rand(*test_shape).astype(var.tensor.dtype))
    return var