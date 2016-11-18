#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
from theano.ifelse import ifelse
from decorations import neural_computation

class Runtime(object):
    """
    Manage runtime variables in deepy.
    """

    def __init__(self):
        self._training_flag = theano.shared(0, name="training_flag")

    @neural_computation
    def iftrain(self, then_branch, else_branch):
        return ifelse(self._training_flag, then_branch, else_branch, name="iftrain")


if "runtime" not in globals():
    runtime = Runtime()