#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy.core.env import EPSILON
from deepy.core import neural_computation

import theano.tensor as T


@neural_computation
def cross_entropy(y, target_index, mask=None, after_softmax=False):
    if y.ndim == 3:
        return cross_entropy_3d(y, target_index, mask, after_softmax=after_softmax)
    else:
        if str(y.owner.op).lower().startswith("softmax"):
            after_softmax = True
        if not after_softmax:
            y = T.nnet.softmax(y)
        return -T.mean(T.log(y)[T.arange(target_index.shape[0]), target_index])

@neural_computation
def cross_entropy_3d(y, target_index, mask=None, after_softmax=False):
    if str(y.owner.op).lower().startswith("softmax"):
        after_softmax = True
    flat_mask = mask.flatten() if mask else 1

    # Softmax
    shape = y.shape
    y_2d = y.reshape((shape[0] * shape[1], shape[2]))
    if after_softmax:
        softmax_tensor = y_2d * (flat_mask[:, None] if mask else 1)
    else:
        if mask:
            penalties = 99. * (1 - flat_mask)
            y_2d -= penalties[:, None]
        softmax_tensor = T.nnet.softmax(y_2d)

    # Get cost
    result_vector = softmax_tensor.flatten()
    target_vector = target_index.flatten()
    target_index_vector = T.arange(target_vector.shape[0]) * shape[-1] + target_vector

    prob_vector = result_vector[target_index_vector]
    prob_vector = T.clip(prob_vector, EPSILON, 1.0 - EPSILON)
    log_prob_vector = - T.log(prob_vector) * flat_mask
    cost = T.sum(log_prob_vector) / T.sum(flat_mask)
    return cost

@neural_computation
def least_squares(y, target):
    err = y - target
    return T.mean((err * err).sum(axis=target.ndim - 1)) / 2

@neural_computation
def accuracy(y, target_index, mask=None):
    if mask:
        target_index = target_index * mask - (1 - mask)
    hits = T.eq(y, target_index)
    if mask:
        return T.sum(hits) / T.sum(mask)
    else:
        return T.mean(hits)

@neural_computation
def error_rate(y, target_index):
    return 1. - accuracy(y, target_index)


