#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging as loggers
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse

from deepy.util.functions import FLOATX, global_rand, make_float_matrices, make_float_vectors
from deepy.util.functions import replace_graph as RG, smart_replace_graph as SRG
from deepy.trainers.optimize import optimize_parameters
from deepy.util import build_activation
from layer import NeuralLayer
from basic_nn import NeuralNetwork


logging = loggers.getLogger(__name__)


class RecurrentLayer(NeuralLayer):

    def __init__(self, size, target_size=-1, activation='sigmoid', noise=0., dropouts=0., update_h0=True, bptt=True,
                 bptt_steps=5, beta=0.0000001, optimization="ADAGRAD"):
        """
        Simple RNN Layer, input x sequence, output y sequence, cost, update parameters.
        Train a RNN without BPTT layers, which means the history_len should be set to 0 for the training data.
        BPTT is conducted once for every piece of data
        :return:
        """
        super(RecurrentLayer, self).__init__(size, activation, noise, dropouts)
        self.learning_rate = 0.1
        self.update_h0 = update_h0
        self.disable_bias = True
        self.bptt = bptt
        self.bptt_steps = bptt_steps
        self.target_size = target_size
        self.optimization = optimization
        self.beta = beta

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

    def _cost_func(self, s ,k_t):
        # return 0.5 * T.sum((s - T.eq(T.arange(s.shape[0]), k_t)) ** 2)
        return -T.log(s[k_t])

    def _recurrent_func(self):

        def recurrent_step(x_t, k_t, h_t):
            h = self._activation_func(T.dot(x_t, self.W_i)+ T.dot(h_t, self.W_r))
            s = self._softmax_func(T.dot(h, self.W_s))

            _cost = self._cost_func(s, k_t)

            g_L_h = T.grad(_cost, h)

            stepping_updates = optimize_parameters([self.W_s], [T.grad(_cost, self.W_s)],
                                                   method=self.optimization, lr=self.learning_rate, beta=self.beta)

            if not self.bptt:
                stepping_updates += optimize_parameters([self.W_i, self.W_r], [T.grad(_cost, self.W_i), T.grad(_cost, self.W_r)],
                                                        method=self.optimization, lr=self.learning_rate, beta=self.beta)

            self._assistive_params.extend([x[0] for x in stepping_updates if x[0] not in {self.W_i, self.W_s, self.W_r}])
            return [h ,s, g_L_h], stepping_updates

        [h_list, s_list, h_errs], updates = theano.scan(fn=recurrent_step, sequences=[self.x, self._vars.k],
                                                outputs_info=[self.h0, None, None], n_steps=self.x.shape[0],
                                                truncate_gradient=self.bptt_steps)


        # BPTT implementation:

        def bptt_step(i, wi, wr, bptt_error, xs, h_list, h_errs):

            # Make virtual graph
            _wi, _wr, _ws = make_float_matrices("_wi", "_wr", "_ws")
            _h, _r, _x, _s, _xz = make_float_vectors("_h", "_r", "_x", "_s", "_xz")

            _z = T.dot(_x, _wi)+ T.dot(_r, _wr)
            _h = self._activation_func(_z)

            _xh = self._activation_func(_xz)

            node_map = {_h: h_list[i], _r: ifelse(T.eq(i, 0), self.h0, h_list[i-1]),
                         _x: xs[i], _wi: self.W_i, _wr: self.W_r}

            # Backpropagate

            g_z_wi = RG(T.grad(T.sum(_z), _wi), node_map)
            g_z_wr = RG(T.grad(T.sum(_z), _wr), node_map)

            error = h_errs[i] + bptt_error

            # Make sure g_h_z only relies on _xh!!!
            g_h_z = SRG(T.grad(T.sum(_xh), _xz), {_xh: h_list[i]})

            g_wi = (g_z_wi * (g_h_z * error))
            g_wr = (g_z_wr * (g_h_z * error))

            # TODO: Updating weights in each step is slow, move it out
            w_updates = OrderedDict(optimize_parameters([wi, wr], [g_wi, g_wr], shapes=[self.W_i, self.W_r],
                                                        method=self.optimization, lr=self.learning_rate, beta=self.beta))

            wi_out = w_updates[wi]
            wr_out = w_updates[wr]
            del w_updates[wr]
            del w_updates[wi]
            self._assistive_params.extend(w_updates.keys())

            bptt_error = T.dot(self.W_r, g_h_z * error)

            return [wi_out, wr_out, bptt_error], w_updates

        [wi_out, wr_out, _], bptt_updates = theano.scan(fn=bptt_step, sequences=[T.arange(self.x.shape[0] - 1, -1, -1)],
                           non_sequences=[self.x, h_list, h_errs], outputs_info=[self.W_i, self.W_r, self.zero_vector])

        if self.bptt:
            updates[self.W_i] = wi_out[-1]
            updates[self.W_r] = wr_out[-1]

        return h_list, s_list, updates + bptt_updates

    def _predict_func(self):

        def predict_step(x_t, h_t):
            if self.disable_bias:
                br = bs = 0
            else:
                br = self.B_r
                bs = self.B_s
            h = self._activation_func(T.dot(x_t, self.W_i)+ T.dot(h_t, self.W_r) + br)
            s = self._softmax_func(T.dot(h, self.W_s) + bs)
            return [h ,s]
        [h_list, s_list], _ = theano.scan(fn=predict_step, sequences=[self.x], outputs_info=[self.h0, None],
                                          n_steps=self.x.shape[0])
        return s_list, [(self.h0, h_list[-1])]

    def _setup_functions(self):
        self._assistive_params = []
        self._activation_func = build_activation(self.activation)
        self._softmax_func = build_activation('softmax')
        self.hidden_func, self.output_func, recurrent_updates = self._recurrent_func()
        self.predict_func, self.predict_updates = self._predict_func()
        self.monitors.append(("hh<0.1", 100 * (abs(self.hidden_func[-1]) < 0.1).mean()))
        self.monitors.append(("hh<0.9", 100 * (abs(self.hidden_func[-1]) < 0.9).mean()))

        self.updates.extend(recurrent_updates.items())
        if self.update_h0:
            self.updates.append((self.h0, ifelse(T.eq(self._vars.k[-1], 0), self.init_h, self.hidden_func[-1])))
        self.params.extend(self._assistive_params)

    def _setup_params(self):
        if self.target_size < 0:
            self.target_size = self.input_n
        self.h0 = theano.shared(value=np.ones((self.output_n,), dtype=FLOATX), name='h_input')
        self.init_h = theano.shared(value=self.h0.get_value(), name='init_h')
        self.zero_vector = theano.shared(value=np.zeros((self.output_n,), dtype=FLOATX), name='zero_h')

        self.W_i, _, self.param_count = self.create_params(self.input_n, self.output_n, "input")
        self.W_r, self.B_r, param_count = self.create_params(self.output_n, self.output_n, "recurrent")
        self.param_count += param_count
        self.W_s, self.B_s, param_count = self.create_params(self.output_n, self.target_size, "softmax")
        self.param_count += param_count

        # Don't register parameters to the weights or bias
        # Update inside the recurrent steps
        self.W = []
        self.B = []
        self.params = [self.W_i, self.W_r, self.W_s]
        if not self.disable_bias:
            self.params += [self.B_r, self.B_s]


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

    def clear_hidden(self):
        self.h0.set_value(np.zeros((self.output_n,), dtype=FLOATX))

    def reset_assistive_params(self):
        for p in self._assistive_params:
            p.set_value(p.get_value() ** 0)


class RecurrentNetwork(NeuralNetwork):

    def __init__(self, config):
        super(RecurrentNetwork, self).__init__(config)
        self._predict_compiled = False
        self.do_reset_grads = True

    def setup_vars(self):
        super(RecurrentNetwork, self).setup_vars()

        # for a classifier, k specifies the correct labels for a given input.
        self.vars.k = T.ivector('k')
        self.inputs.append(self.vars.k)

    @property
    def cost(self):
        return -T.mean(T.log(self.vars.y)[T.arange(self.vars.k.shape[0]), self.vars.k])

    @property
    def errors(self):
        return 100 * T.mean(T.neq(T.argmax(self.vars.y, axis=1), self.vars.k))

    @property
    def monitors(self):
        yield 'err', self.errors
        for name, exp in self.special_monitors:
            yield name, exp

    def _compile(self):
        if not self._predict_compiled:
            rnn_layer = self.layers[0]
            self._predict_rnn = theano.function([self.vars.x], [rnn_layer.predict_func], updates=rnn_layer.predict_updates)
            self._predict_compiled = True

    def classify(self, x):
        self._compile()
        return np.argmax(self._predict_rnn(x)[0], axis=1)

    def get_probs(self, x, k):
        self._compile()
        k = np.array(k)
        return self._predict_rnn(x)[0][np.arange(k.shape[0]), k]

    def clear_hidden(self):
        rnn_layer = self.layers[0]
        rnn_layer.clear_hidden()

    def iteration_callback(self):
        rnn_layer = self.layers[0]
        if self.do_reset_grads:
            rnn_layer.reset_assistive_params()