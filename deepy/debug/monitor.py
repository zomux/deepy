#!/usr/bin/env python
# -*- coding: utf-8 -*-
import theano
from theano import tensor as T

from deepy.core.env import FLOATX

def monitor(var, name=""):
    return monitor_tensor(var.tensor, name=name)


def monitor_tensor(value, name="", disabled=False):
    if disabled:
        return T.sum(value)*0
    else:
        val = T.sum(theano.printing.Print(name)(value))*T.constant(0.0000001, dtype=FLOATX)
        return T.cast(val, FLOATX)