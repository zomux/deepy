#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging as loggers

import numpy as np
import theano
import theano.tensor as T

from deepy.util.functions import FLOATX, global_rand
from deepy.util import build_activation


logging = loggers.getLogger(__name__)


class NeuralLayer(object):

    def __init__(self, size, activation='tanh', noise=0., dropouts=0., shared_bias=None, disable_bias=True):
        """
        Create a neural layer.
        :return:
        """
        self.activation = activation
        self.size = size
        self.output_n = size
        self.connected = False
        self.noise = noise
        self.dropouts = dropouts
        self.shared_bias = shared_bias
        self.disable_bias = disable_bias
        self.updates = []
        self.learning_updates = []
        self.W = []
        self.B = []
        self.params = []
        self.monitors = []
        self.inputs = []
        self.param_count = 0

    def connect(self, config, vars, x, input_n, id="UNKNOWN"):
        """
        Connect to a network
        :type config: deepy.conf.NetworkConfig
        :type vars: deepy.functions.VarMap
        :return:
        """
        self._config = config
        self._vars = vars
        self.input_n = input_n
        self.id = id
        self.x = x
        self._setup_params()
        self._setup_functions()
        self.connected = True

    def _setup_functions(self):
        if self.shared_bias:
            self._vars.update_if_not_existing(self.shared_bias, self.B)
        bias = self.B if not self.shared_bias else self._vars.get(self.shared_bias)
        if self.disable_bias:
            bias = 0

        self._activation_func = build_activation(self.activation)
        self.preact_func = T.dot(self.x, self.W) + bias
        self.output_func = build_activation.add_noise(
                self._activation_func(self.preact_func),
                self.noise,
                self.dropouts)

    def _setup_params(self):
        self.W = self.create_weight(self.input_n, self.output_n, self.id)
        self.B = self.create_bias(self.output_n, self.id)
        if self.disable_bias:
            self.B = []

    def create_weight(self, input_n=1, output_n=1, suffix="", sparse=None, scale=None, shape=None):
        # ws = np.asarray(global_rand.uniform(low=-np.sqrt(6. / (input_n + output_n)),
        #                           high=np.sqrt(6. / (input_n + output_n)),
        #                           size=(input_n, output_n)))
        # if self.activation == 'sigmoid':
        #     ws *= 4
        # if sparse is not None:
        #     ws *= np.random.binomial(n=1, p=sparse, size=(input_n, output_n))

        adjust_weights = False
        if not shape:
            shape = (input_n, output_n)
            adjust_weights = True

        if not scale:
            scale = np.sqrt(6. / sum(shape))
            if self.activation == 'sigmoid':
                scale *= 4

        ws = np.asarray(global_rand.uniform(low=-scale, high=scale, size=shape))

        # Adjust weights
        if adjust_weights:
            norm = np.sqrt((ws**2).sum())
            ws = scale * ws / norm
            _, v, _ = np.linalg.svd(ws)
            ws = scale * ws / v[0]

        weight = theano.shared(ws.astype(FLOATX), name='W_{}'.format(suffix))

        self.param_count += np.prod(shape)
        logging.info('create weight W_%s: %s', suffix, str(shape))
        return weight

    def create_bias(self, output_n=1, suffix="", value=0., shape=None):
        if not shape:
            shape = (output_n, )
        bs =  np.ones(shape)
        bs *= value
        bias = theano.shared(bs.astype(FLOATX), name='B_{}'.format(suffix))
        self.param_count += np.prod(shape)
        logging.info('create bias B_%s: %s', suffix, str(shape))
        return bias

    def create_vector(self, n, name, dtype=FLOATX):
        bs =  np.zeros(n)
        v = theano.shared(bs.astype(dtype), name='{}'.format(name))

        logging.info('create vector %s: %d', name, n)
        return v

    def create_matrix(self, m, n, name):

        matrix = theano.shared(np.zeros((m, n)).astype(FLOATX), name=name)

        logging.info('create matrix %s: %d x %d', name, m, n)
        return matrix

    def callback_forward_propagation(self):
        pass