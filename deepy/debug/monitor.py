#!/usr/bin/env python
# -*- coding: utf-8 -*-
import theano
from theano import tensor as T

from deepy.core.env import FLOATX


def monitor_var_sum(value, name="", disabled=False):
    if disabled:
        return T.sum(value)*0
    else:
        val = T.sum(theano.printing.Print(name)(value))*T.constant(0.0000001, dtype=FLOATX)
        return T.cast(val, FLOATX)


def monitor_var(value, name="", disabled=False):
    if disabled:
        return value
    else:
        return theano.printing.Print(name)(value)