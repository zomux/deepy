#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano.tensor as T
import inspect

def wrap_core(func, train_config, *args):
    spec = inspect.getargspec(func)
    params = spec.args[-len(spec.defaults):]
    default_values = spec.defaults
    kwargs = dict(zip(params, default_values))
    if train_config:
        for param, default in zip(params, default_values):
            config_val = train_config.get(param, default)
            kwargs[param] = config_val
    return list(func(*args, **kwargs))


def multiple_l2_norm(tensors):
    """
    Get the L2 norm of multiple tensors.
    This function is taken from blocks.
    """
    flattened = [T.as_tensor_variable(t).flatten() for t in tensors]
    flattened = [(t if t.ndim > 0 else t.dimshuffle('x'))
                 for t in flattened]
    joined = T.join(0, *flattened)
    return T.sqrt(T.sqr(joined).sum())

