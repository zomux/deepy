#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import numpy as np

def momentum_core(params, gradients, momentum=0.9, learning_rate=0.01):
    """
    Momentum SGD optimization core.
    """
    free_parameters = []
    updates = []
    for param, grad in zip(params, gradients):
            delta = learning_rate * grad
            velocity = theano.shared(np.zeros_like(param.get_value()), name=param.name + '_vel')
            updates.append((velocity, momentum * velocity - delta))
            updates.append((param, param + velocity))
            free_parameters.append(velocity)
    return updates, free_parameters

