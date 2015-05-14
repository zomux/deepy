#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict
import inspect
import numpy as np
import theano
from theano import tensor as T
from deepy.trainers.optimize import logging
from deepy.utils import FLOATX


def ada_family_core(params, gparams, learning_rate = 0.01, eps= 1e-6, rho=0.95, method="ADADELTA",
                        beta=0.0, gsum_regularization = 0.0001):
    """
    Optimize by SGD, AdaGrad, or AdaDelta.
    """

    _, _, _, args = inspect.getargvalues(inspect.currentframe())
    logging.info("ada_family_core: %s" % str(args.items()))

    if method == "FINETUNING_ADAGRAD":
        method = "ADAGRAD"
        gsum_regularization = 0

    oneMinusBeta = 1 - beta

    gsums   = [theano.shared(np.zeros_like(param.get_value(borrow=True), dtype=FLOATX), name="gsum_%s" % param.name) if (method == 'ADADELTA' or method == 'ADAGRAD') else None for param in params]
    xsums   = [theano.shared(np.zeros_like(param.get_value(borrow=True), dtype=FLOATX), name="xsum_%s" % param.name) if method == 'ADADELTA' else None for param in params]

    # Fix for AdaGrad, init gsum to 1
    if method == 'ADAGRAD':
        for gsum in gsums:
            gsum.set_value(gsum.get_value() ** 0)

    updates = OrderedDict()

    for gparam, param, gsum, xsum in zip(gparams, params, gsums, xsums):

        if method == 'ADADELTA':
            updates[gsum] = rho * gsum + (1. - rho) * (gparam **2)
            dparam = -T.sqrt((xsum + eps) / (updates[gsum] + eps)) * gparam
            updates[xsum] =rho * xsum + (1. - rho) * (dparam **2)
            updates[param] = param * oneMinusBeta + dparam
        elif method == 'ADAGRAD':
            updates[gsum] = gsum + (gparam **2) - gsum_regularization * gsum
            updates[param] =  param * oneMinusBeta - learning_rate * (gparam / (T.sqrt(updates[gsum] + eps)))
        else:
            updates[param] = param * oneMinusBeta - gparam * learning_rate

    return updates.items()