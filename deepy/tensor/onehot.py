#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from theano import tensor as T

from deepy.core import env


def onehot_tensor(i_matrix, vocab_size):
    """
    # batch x time
    """
    dim0, dim1 = i_matrix.shape
    i_vector = i_matrix.reshape((-1,))
    hot_matrix = T.extra_ops.to_one_hot(i_vector, vocab_size).reshape((dim0, dim1, vocab_size))
    return hot_matrix


def onehot(size, eye):
    return np.eye(1, size, eye, dtype=env.FLOATX)[0]