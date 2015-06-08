#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy.utils import CrossEntropyCost, EPSILON
import theano.tensor as T

class LMCost(CrossEntropyCost):

    def get(self):
        y = T.clip(self.result_tensor, EPSILON, 1.0 - EPSILON)
        y = y.reshape((-1, y.shape[-1]))
        k = self.index_tensor.reshape((-1,))
        return -T.mean(T.log2(y[T.arange(k.shape[0]), k]))