#!/usr/bin/env python
# -*- coding: utf-8 -*-

from theano.ifelse import ifelse as theano_ifelse

from deepy.layers.concat import Concatenate
from deepy.core.decorations import neural_computation

@neural_computation
def concatenate(vars, axis=-1):
    """
    A utility function of concatenate.
    """
    concat_var = Concatenate(axis=axis).compute(*vars)
    if axis == -1 or axis == vars[0].tensor.ndim - 1:
        concat_var.output_dim = sum([x.output_dim for x in vars], 0)
    return concat_var

@neural_computation
def ifelse(condition, then_branch, else_branch):
    return theano_ifelse(condition, then_branch, else_branch)


def apply(func, *args, **kwargs):
    from deepy.core.neural_var import NeuralVariable
    dim = kwargs['dim'] if 'dim' in kwargs else args[0].dim()
    return NeuralVariable(func(*[x.tensor for x in args]), dim)
