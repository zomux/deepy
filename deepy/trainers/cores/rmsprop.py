#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np

def rmsprop_core(params, gradients, momentum=0.9, learning_rate=0.01):
    """
    RMSPROP optimization core.
    """
    for param, grad in zip(params, gradients):
            rms_ = theano.shared(np.zeros_like(param.get_value()), name=param.name + '_rms')
            rms = momentum * rms_ + (1 - momentum) * grad * grad
            yield rms_, rms
            yield param, param - learning_rate * grad / T.sqrt(rms + 1e-8)