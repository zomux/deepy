#!/usr/bin/env python
# -*- coding: utf-8 -*-


from collections import OrderedDict
import inspect
import logging as loggers

import theano
import theano.tensor as T
import numpy as np

from deepy.util.functions import FLOATX
from deepy.trainers.util import wrap_core


logging = loggers.getLogger(__name__)

def optimize(params, gradients, method="ADAM", clip=True, max_norm=5.0, config=None):
    """
    General optimization function for Theano.
    """
    func = None

    if max_norm and clip:
        clipped_gradients = []
        norm_constant = T.constant(max_norm, dtype=FLOATX)
        for g in gradients:
            grad_norm = g.norm(L=2)
            g = (T.minimum(norm_constant, grad_norm)/ grad_norm) * g
            clipped_gradients.append(g)
        gradients = clipped_gradients

    if method == "ADAM":
        from adam import adam_core
        func = adam_core

    if not func:
        raise NotImplementedError("method '%s' is not supported" % method)

    logging.info("optimize method=%s parameters=%s" % (method, str(params)))

    return wrap_core(func, config, params, gradients)

def optimize_parameters(params, gparams, shapes=None, max_norm = 5.0, lr = 0.01, eps= 1e-6, rho=0.95, method="ADADELTA",
                        beta=0.0, gsum_regularization = 0, weight_l2 = 0, clip = True):
    """
    Optimize by SGD, AdaGrad, or AdaDelta.
    Returns the shared variables for the gradient caches,
    and the updates dictionary for compilation by a
    theano function.

    :param params: parameters to optimize
    :param gparams: gradients
    :param max_norm: cap on excess gradients
    :param lr: base learning rate for adagrad and SGD
    :param eps: numerical stability value to not divide by zero sometimes
    :param rho: adadelta hyperparameter
    :param method: 'ADAGRAD', 'ADADELTA', or 'SGD'

    :returns updates: the updates to pass to theano function
    :returns gsums: gradient caches for Adagrad and AdaDelta
    :returns xsums: gradient caches for AdaDelta only
    :returns lr: theano shared : learning rate
    :returns max_norm theano_shared : normalizing clipping value for excessive gradients (exploding)
    """

    _, _, _, args = inspect.getargvalues(inspect.currentframe())
    logging.info("optimize params: %s" % str(args.items()))

    if method == "FINETUNING_ADAGRAD":
        method = "ADAGRAD"
        gsum_regularization = 0

    # lr = theano.shared(np.float64(lr).astype(FLOATX))
    if not shapes:
        shapes = params
    # eps = np.float64(eps).astype(FLOATX)
    # rho = np.float64(rho).astype(FLOATX)
    # if max_norm is not None and max_norm is not False:
    #     max_norm = theano.shared(np.float64(max_norm).astype(FLOATX))
    oneMinusBeta = 1 - beta


    gsums   = [theano.shared(np.zeros_like(param.get_value(borrow=True), dtype=FLOATX), name="gsum_%s" % param.name) if (method == 'ADADELTA' or method == 'ADAGRAD') else None for param in shapes]
    xsums   = [theano.shared(np.zeros_like(param.get_value(borrow=True), dtype=FLOATX), name="xsum_%s" % param.name) if method == 'ADADELTA' else None for param in shapes]

    # Fix for AdaGrad, init gsum to 1
    if method == 'ADAGRAD':
        for gsum in gsums:
            gsum.set_value(gsum.get_value() ** 0)

    updates = OrderedDict()

    for gparam, param, gsum, xsum in zip(gparams, params, gsums, xsums):
        # clip gradients if they get too big
        if max_norm and clip:
            grad_norm = gparam.norm(L=2)
            gparam = (T.minimum(T.constant(max_norm, dtype=FLOATX), grad_norm)/ grad_norm) * gparam

        if method == 'ADADELTA':
            if weight_l2 > 0:
                gparam += (2 * weight_l2 * param)
            updates[gsum] = rho * gsum + (1. - rho) * (gparam **2)
            dparam = -T.sqrt((xsum + eps) / (updates[gsum] + eps)) * gparam
            updates[xsum] =rho * xsum + (1. - rho) * (dparam **2)
            updates[param] = param * oneMinusBeta + dparam
        elif method == 'ADAGRAD':
            # updates[gsum] =  gsum + (gparam ** 2)
            updates[gsum] = gsum + (gparam **2) - gsum_regularization * gsum
            updates[param] =  param * oneMinusBeta - lr * (gparam / (T.sqrt(updates[gsum] + eps)) + (2 * weight_l2 * param))
        else:
            updates[param] = param * oneMinusBeta - (gparam  + (2 * weight_l2 * param)) * lr

    return updates.items()
    #return updates, gsums, xsums, lr, max_normjnn

def gradient_interface(params, **kwargs):
    gs = [T.matrix() if p.ndim == 2 else T.vector() for p in params]
    updates = optimize_parameters(params, gs, **kwargs)
    return theano.function(gs, [], updates=updates)