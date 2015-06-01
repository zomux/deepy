#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano.tensor as T

class Cost(object):

    def get(self):
        return NotImplementedError()

class CrossEntropyCost(Cost):

    def __init__(self, result_tensor, index_tensor):
        self.result_tensor = result_tensor
        self.index_tensor = index_tensor

    def get(self):
        return -T.mean(T.log(self.result_tensor)[T.arange(self.index_tensor.shape[0]), self.index_tensor])

class RegressionCost(Cost):

    def __init__(self, result_tensor, target_tensor):
        self.result_tensor = result_tensor
        self.target_tensor = target_tensor

    def get(self):
        err = self.result_tensor - self.target_tensor
        return T.mean((err * err).sum(axis=self.target_tensor.ndim - 1)) / 2

class AutoEncoderCost(Cost):

    def __init__(self, result_tensor, target_tensor):
        self.result_tensor = result_tensor
        self.target_tensor = target_tensor

    def get(self):
        return T.sum((self.result_tensor - self.target_tensor)**2)

class ErrorRateCost(Cost):

    def __init__(self, result_tensor, index_tensor):
        self.result_tensor = result_tensor
        self.index_tensor = index_tensor

    def get(self):
        return 100 * T.mean(T.neq(T.argmax(self.result_tensor, axis=1), self.index_tensor))