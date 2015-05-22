#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging as loggers

import numpy as np
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

from deepy.utils import build_activation, UniformInitializer
from deepy.layers.layer import NeuralLayer


logging = loggers.getLogger(__name__)


class Convolution(NeuralLayer):
    """
    Convolution layer with max-pooling.
    """

    def __init__(self, filter_shape, pool_size=(2, 2),
                 reshape_input=False, border_mode="valid", flatten_output=False,
                 disable_pooling=False,  activation='linear', init=None):
        super(Convolution, self).__init__("convolution")
        self.filter_shape = filter_shape
        self.output_dim = filter_shape[0]
        self.pool_size = pool_size
        self.reshape_input = reshape_input
        self.flatten_output = flatten_output
        self.activation = activation
        self.disable_pooling = disable_pooling
        self.border_mode = border_mode
        self.initializer = init if init else self._default_initializer()

    def setup(self):
        self._setup_params()
        self._setup_functions()

    def output(self, x):
        if self.reshape_input:
            img_width = T.cast(T.sqrt(x.shape[1]), "int32")
            x = x.reshape((x.shape[0], 1, img_width, img_width), ndim=4)

        conv_out = conv.conv2d(
            input=x,
            filters=self.W_conv,
            filter_shape=self.filter_shape,
            image_shape=None,
            border_mode=self.border_mode
        )

        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=self.pool_size,
            ignore_border=True
        )

        if self.disable_pooling:
            pooled_out = conv_out

        output = self._activation_func(pooled_out + self.B_conv.dimshuffle('x', 0, 'x', 'x'))

        if self.flatten_output:
            output = output.flatten(2)
        return output

    def _setup_functions(self):
        self._activation_func = build_activation(self.activation)

    def _setup_params(self):
        self.W_conv = self.create_weight(suffix="conv", initializer=self.initializer, shape=self.filter_shape)
        self.B_conv = self.create_bias(self.filter_shape[0], suffix="conv")

        self.register_parameters(self.W_conv, self.B_conv)

    def _default_initializer(self):
        fan_in = np.prod(self.filter_shape[1:])
        fan_out = (self.filter_shape[0] * np.prod(self.filter_shape[2:]) /
                   np.prod(self.pool_size))
        weight_scale = np.sqrt(6. / (fan_in + fan_out))
        return UniformInitializer(scale=weight_scale)