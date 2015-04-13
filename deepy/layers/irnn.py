#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import RNN
from deepy.util import GaussianInitializer, IdentityInitializer

class IRNN(RNN):
    """
    The implementation of http://arxiv.org/abs/1504.00941 .
    RNN with weight initialization using identity matrix.
    """

    def __init__(self, hidden_size, output_size=None, input_type="sequence", output_type="last_hidden",
                 activation="tanh", weight_scale=1, steps=None):
        super(IRNN, self).__init__(hidden_size, output_size=output_size,
                                   input_type=input_type, output_type=output_type,
                                   hidden_activation="relu",
                                   hidden_initializer=IdentityInitializer(scale=weight_scale),
                                   initializer=GaussianInitializer(deviation=0.001))