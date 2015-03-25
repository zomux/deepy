#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect

def wrap_core(func, train_config, *args):
    spec = inspect.getargspec(func)
    params = spec.args[-len(spec.defaults):]
    default_values = spec.defaults
    kwargs = dict(zip(params, default_values))
    if train_config:
        for p in params:
            config_val = getattr(train_config, p)
            if config_val != None:
                kwargs[p] = config_val
    return func(*args, **kwargs)