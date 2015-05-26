#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging as loggers

import theano
import theano.tensor as T
from theano.ifelse import ifelse

from deepy.utils import FLOATX, dim_to_var, EPSILON
from deepy.trainers.util import wrap_core


logging = loggers.getLogger(__name__)

def optimize_updates(params, gradients, config=None, shapes=None):
    """
    General optimization function for Theano.
    Parameters:
        params - parameters
        gradients - gradients
        config - training config
    Returns:
        Theano updates
    :type config: deepy.TrainerConfig
    """
    # Clipping
    if config:
        clip_value = config.get("max_norm", 5.0)
        clip = config.get("gradient_clipping", None)

        if clip_value and clip:
            clip_constant = T.constant(clip_value, dtype=FLOATX)
            clipped_gradients = []
            for g in gradients:
                grad_norm = g.norm(L=1) if clip == "l1" else g.norm(L=2)
                if config.gradient_tolerance:
                    g = ifelse(grad_norm > config.gradient_tolerance, T.zeros_like(g) + EPSILON, g)
                multiplier = ifelse(grad_norm < clip_constant,
                                    T.constant(1., dtype=FLOATX), clip_constant / (grad_norm + EPSILON))
                g = multiplier * g
                clipped_gradients.append(g)
            gradients = clipped_gradients
    # Regularization
    if config and config.weight_l2:
        regularized_gradients = []
        for param, grad in zip(params, gradients):
            grad = grad + (2 * config.weight_l2 * param)
            regularized_gradients.append(grad)
        gradients = regularized_gradients

    # Avoid nan
    if config.avoid_nan:
        logging.info("avoid NaN gradients")
        new_gradients = []
        for grad in gradients:
            new_grad = ifelse(T.isnan(grad.max()), T.zeros_like(grad) + EPSILON, grad)
            new_gradients.append(new_grad)
        gradients = new_gradients


    # Find method
    method = "SGD"
    if config:
        method = config.get("method", method).upper()
    # Get Function
    func = None
    if method in ["SGD", "ADAGRAD", "ADADELTA", "FINETUNING_ADAGRAD"]:
        from cores.ada_family import ada_family_core
        func = ada_family_core
    elif method == "ADAM":
        from cores.adam import adam_core
        func = adam_core
    elif method == "RMSPROP":
        from cores.rmsprop import rmsprop_core
        func = rmsprop_core
    elif method == "MOMENTUM":
        from cores.momentum import momentum_core
        func = momentum_core

    if not func:
        raise NotImplementedError("method '%s' is not supported" % method)

    logging.info("optimize method=%s parameters=%s" % (method, str(params)))

    updates = wrap_core(func, config, params, gradients)
    # Weight bound
    if config.weight_bound:
        logging.info("apply weight bound of %.2f" % config.weight_bound)
        new_updates = []
        for param, update_value in updates:
            bounded_value = (update_value * (T.abs_(update_value) <= config.weight_bound) +
                             config.weight_bound * (update_value > config.weight_bound) +
                             -config.weight_bound * (update_value < -config.weight_bound))
            new_updates.append((param, bounded_value))
        updates = new_updates
    return updates

def optimize_function(params, config=None):
    """
    Create a optimizing function receives gradients.
    Parameters:
        params - parameters
        config - training configuration
    Returns:
        updating function receives gradients
    """
    gs = [dim_to_var(p.ndim) for p in params]
    updates = optimize_updates(params, gs, config)
    return theano.function(gs, [], updates=updates)