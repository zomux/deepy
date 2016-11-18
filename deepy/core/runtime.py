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
        """
        Execute `then_branch` when training.
        """
        return ifelse(self._training_flag, then_branch, else_branch, name="iftrain")

    def switch_training(self, flag):
        """
        Switch training mode.
        :param flag: switch on training mode when flag is True.
        """
        if flag:
            self._training_flag.set_value(1)
        else:
            self._training_flag.set_value(0)


if "runtime" not in globals():
    runtime = Runtime()