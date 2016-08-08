#!/usr/bin/env python
# -*- coding: utf-8 -*-

from concat import Concatenate
from theano.ifelse import ifelse as theano_ifelse
from deepy.utils.decorations import neural_computation

def concatenate(vars, axis=-1):
    """
    A utility function of concatenate.
    """
    return Concatenate(axis=axis).compute(*vars)

@neural_computation
def ifelse(condition, then_branch, else_branch):
    return theano_ifelse(condition, then_branch, else_branch)

