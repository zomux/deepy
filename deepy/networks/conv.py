#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging as loggers

import numpy as np
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

from deepy.util import build_activation
from deepy.networks.layer import NeuralLayer


logging = loggers.getLogger(__name__)


class ConvPoolLayer(NeuralLayer):

    def __init__(self, filter_shape, pool_size=(2, 2), output_size=0,
                 reshape_input=False, flatten_output=False,
                 activation='tanh', noise=0., dropouts=0.):
        super(ConvPoolLayer, self).__init__(output_size, activation, noise, dropouts)
        self.filter_shape = filter_shape
        self.pool_size = pool_size
        self.reshape_input = reshape_input
        self.flatten_output = flatten_output

    def connect(self, config, vars, x, input_n, id="UNKNOWN"):
        self._config = config
        self._vars = vars
        self.input_n = input_n
        self.id = id
        self.x = x
        self._setup_params()
        self._setup_functions()
        self.connected = True

    def _output_func(self):
        if self.reshape_input:
            img_width = T.cast(T.sqrt(self.x.shape[1]), "int32")
            self.x = self.x.reshape((self.x.shape[0], 1, img_width, img_width), ndim=4)

        conv_out = conv.conv2d(
            input=self.x,
            filters=self.W_conv,
            filter_shape=self.filter_shape,
            image_shape=None
        )

        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=self.pool_size,
            ignore_border=True
        )

        output = self._activation_func(pooled_out + self.B_conv.dimshuffle('x', 0, 'x', 'x'))

        if self.flatten_output:
            output = output.flatten(2)
        return output

    def _setup_functions(self):
        self._assistive_params = []
        self._activation_func = build_activation(self.activation)
        self.output_func = self._output_func()

    def _setup_params(self):

        fan_in = np.prod(self.filter_shape[1:])
        fan_out = (self.filter_shape[0] * np.prod(self.filter_shape[2:]) /
                   np.prod(self.pool_size))
        weight_scale = np.sqrt(6. / (fan_in + fan_out))

        self.W_conv = self.create_weight(suffix="conv", scale=weight_scale, shape=self.filter_shape)
        self.B_conv = self.create_bias(self.filter_shape[0], suffix="conv")

        self.W = [self.W_conv]
        self.B = [self.B_conv]
        self.params = []