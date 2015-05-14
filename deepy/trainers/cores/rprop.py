#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano.tensor as T
import theano

def rprop_core(params, gradients, rprop_increase=1.01, rprop_decrease=0.99, rprop_min_step=0, rprop_max_step=100,
               learning_rate=0.01):
    """
    Rprop optimizer.
    See http://sci2s.ugr.es/keel/pdf/algorithm/articulo/2003-Neuro-Igel-IRprop+.pdf.
    """
    for param, grad in zip(params, gradients):
        grad_tm1 = theano.shared(np.zeros_like(param.get_value()), name=param.name + '_grad')
        step_tm1 = theano.shared(np.zeros_like(param.get_value()) + learning_rate, name=param.name+ '_step')

        test = grad * grad_tm1
        same = T.gt(test, 0)
        diff = T.lt(test, 0)
        step = T.minimum(rprop_max_step, T.maximum(rprop_min_step, step_tm1 * (
            T.eq(test, 0) +
            same * rprop_increase +
            diff * rprop_decrease)))
        grad = grad - diff * grad
        yield param, param - T.sgn(grad) * step
        yield grad_tm1, grad
        yield step_tm1, step
