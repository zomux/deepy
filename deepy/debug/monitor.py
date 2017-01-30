#!/usr/bin/env python
# -*- coding: utf-8 -*-
import theano
from theano import tensor as T

from deepy.core.env import FLOATX, EPSILON
from collections import defaultdict


if "stack" not in globals():
    stack = defaultdict(list)


class FuncBreakpointOp(theano.Op):

    view_map = {0: [0]}

    __props__ = ('fn',)

    def __init__(self, fn=None):
        self.fn = fn

    def make_node(self, xin):
        xout = xin.type.make_variable()
        return theano.Apply(op=self, inputs=[xin], outputs=[xout])

    def perform(self, node, inputs, output_storage, **kwargs):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin
        if self.fn is not None:
            self.fn(xin)

    def grad(self, input, output_gradients):
        return output_gradients

    def R_op(self, inputs, eval_points):
        return [x for x in eval_points]

    def __setstate__(self, dct):
        dct.setdefault('fn', self.fn)
        self.__dict__.update(dct)

    def c_code_cache_version(self):
        return (1,)


class RecordOp(FuncBreakpointOp):

    __props__ = ('key',)

    def __init__(self, key="general"):
        self.key = key

    def perform(self, node, inputs, output_storage, **kwargs):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin
        stack[self.key].append(xin)


def monitor(var, name=""):
    return monitor_tensor(var.tensor, name=name)


def monitor_tensor(value, name="", disabled=False):
    if disabled:
        return T.sum(value)*0
    else:
        val = T.sum(theano.printing.Print(name)(value)) * EPSILON
        return T.cast(val, FLOATX)


def breakpoint(value, func):
    return breakpoint_tensor(value.tensor, func)


def breakpoint_tensor(value, func):
    val = T.sum(FuncBreakpointOp(func)(value)) * EPSILON
    return T.cast(val, FLOATX)


def record(value, key="general"):
    return record_tensor(value.tensor, key)


def record_tensor(value, key="general"):
    val = T.sum(RecordOp(key)(value)) * EPSILON
    return T.cast(val, FLOATX)


