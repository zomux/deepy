#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from functions import FLOATX

def create_var(theano_tensor, dim=0, test_shape=None, test_dtype=FLOATX):
    """
    Wrap a Theano tensor into the variable for defining neural network.
    :param dim: last dimension of tensor, 0 indicates that the last dimension is flexible
    :rtype: TensorVar
    """
    from deepy.layers.variable import NeuralVar
    var = NeuralVar(dim, theano_tensor)
    if test_shape:
        var.set_test_value(np.random.rand(*test_shape).astype(test_dtype))
    return var