#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano

def check_test_values():
    theano.config.compute_test_value = 'warn'

def set_trace():
    try:
        from ipdb import set_trace as _set_trace
    except ImportError as e:
        from pdb import set_trace as _set_trace
    _set_trace()