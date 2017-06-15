#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np


def nag_core(params, gradients, momentum=0.99, learning_rate=0.25):
    """
    Momentum SGD optimization core.
    """
    free_parameters = []
    updates = []
    for param, grad in zip(params, gradients):
            step = learning_rate * grad
            m = theano.shared(np.zeros_like(param.get_value()), name=param.name + '_m')
            v = momentum * m - step
            updates.append((m, v))
            updates.append((param, param + momentum * v - step))
            free_parameters.append(m)
    return updates, free_parameters
