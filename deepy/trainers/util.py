#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
