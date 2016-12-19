#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano

def check_test_values():
    theano.config.compute_test_value = 'warn'
