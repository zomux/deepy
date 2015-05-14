#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np

def nag_core(params, J, momentum=0.9, learning_rate=0.01):
    """
    Nesterov's Accelerated Gradient (NAG).
    See http://www.cs.toronto.edu/~fritz/absps/momentum.pdf .
    Still unfinished
    """
    # TODO: this requires some refractorings.
    for param in params:
        step = theano.shared(np.zeros_like(param.get_value()), name=param.name + '_step')
        velocity = theano.shared(np.zeros_like(param.get_value()), name=param.name + '_vel')
        yield step, momentum * velocity
        yield param, param + step
        yield velocity, step - learning_rate * T.grad(J, param)
        yield param, param + velocity - step