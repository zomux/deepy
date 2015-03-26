#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging as loggers

import numpy as np
import theano
import theano.tensor as T

from deepy.util.functions import FLOATX, global_rand
from deepy.util import build_activation
from deepy.networks.layer import NeuralLayer
from basic_nn import NeuralNetwork


logging = loggers.getLogger(__name__)


class SimpleRNNLayer(NeuralLayer):

    def __init__(self, size, target_size=-1, activation='sigmoid', noise=0., dropouts=0.):
        """
        Simple RNN Layer, input x sequence, output y sequence, cost, update parameters.
        Train a RNN without BPTT layers, which means the history_len should be set to 0 for the training data.
        :return:
        """
        super(SimpleRNNLayer, self).__init__(size, activation, noise, dropouts)
        self.target_size = target_size

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

    def _recurrent_step(self, x_t, h_t):
        h = self._activation_func(T.dot(x_t, self.W_i)+ T.dot(h_t, self.W_r))
        s = self._softmax_func(T.dot(h, self.W_s))
        return [h ,s]

    def _recurrent_func(self):

        [h_list, s_list], _ = theano.scan(fn=self._recurrent_step, sequences=self.x, outputs_info=[self.h0, None],
                                          n_steps=self.x.shape[0])

        return h_list, s_list

    def _setup_functions(self):
        self._activation_func = build_activation(self.activation)
        self._softmax_func = build_activation('softmax')
        self.hidden_func, self.output_func = self._recurrent_func()
        self.monitors.append(("h<0.1", 100 * (abs(self.hidden_func[-1]) < 0.1).mean()))
        self.monitors.append(("h<0.9", 100 * (abs(self.hidden_func[-1]) < 0.9).mean()))
        # self.updates.append((self.h0, self.hidden_func[-1]))

    def _setup_params(self):
        if self.target_size < 0:
            self.target_size = self.input_n

        self.h0 = theano.shared(value=np.zeros((self.output_n,), dtype=FLOATX), name='h_input')

        self.W_i, _, self.param_count = self.create_params(self.input_n, self.output_n, "input")
        self.W_r, self.B_r, param_count = self.create_params(self.output_n, self.output_n, "recurrent")
        self.param_count += param_count
        self.W_s, self.B_s, param_count = self.create_params(self.output_n, self.target_size, "softmax")
        self.param_count += param_count

        # Don't register parameters to the whole network
        self.W = [self.W_i, self.W_r, self.W_s]
        self.B = []
        # self.B = [self.B_r, self.B_s]
        self._extra_params()

    def _extra_params(self):
        pass


    def create_params(self, input_n, output_n, suffix, sparse=None):
        # arr = np.random.randn(input_n, output_n) / np.sqrt(input_n + output_n)
        ws = np.asarray(global_rand.uniform(low=-np.sqrt(6. / (input_n + output_n)),
                                  high=np.sqrt(6. / (input_n + output_n)),
                                  size=(input_n, output_n)))
        if self.activation == 'sigmoid':
            ws *= 4
        if sparse is not None:
            ws *= np.random.binomial(n=1, p=sparse, size=(input_n, output_n))
        weight = theano.shared(ws.astype(FLOATX), name='W_{}'.format(suffix))

        bs =  np.zeros(output_n)
        bias = theano.shared(bs.astype(FLOATX), name='b_{}'.format(suffix))
        logging.info('weights for layer %s: %s x %s', suffix, input_n, output_n)
        return weight, bias, (input_n + 1) * output_n


    def reset_hidden_layer(self):
        self.h0.set_value(np.zeros((self.output_n,), dtype=FLOATX), borrow=True)


class SimpleRNN(NeuralNetwork):
    '''A classifier attempts to match a 1-hot target output.'''

    def __init__(self, config):
        super(SimpleRNN, self).__init__(config)

    def setup_vars(self):
        super(SimpleRNN, self).setup_vars()

        # for a classifier, k specifies the correct labels for a given input.
        self.vars.k = T.ivector('k')
        self.inputs.append(self.vars.k)
        self.target_inputs.append(self.vars.k)


    @property
    def cost(self):
        return -T.log(self.vars.y[-1][self.vars.k[-1]])

    @property
    def errors(self):
        return T.neq(T.argmax(self.vars.y[-1]), self.vars.k[-1])

    @property
    def monitors(self):
        yield 'err', self.errors
        for name, exp in self.special_monitors:
            yield name, exp

    def classify(self, *x):
        return self.predict(*x).argmax(axis=1)

    def get_top_prob(self, *x):
        return self.predict(*x).max(axis=1)