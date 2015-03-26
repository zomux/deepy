#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging as loggers

import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse

from deepy.util.functions import FLOATX, make_float_vectors
from deepy.util.functions import smart_replace_graph as SRG
from deepy.trainers.optimize import optimize_parameters
from deepy.trainers.minibatch_optimizer import MiniBatchOptimizer
from deepy.util import build_activation
from deepy.networks.layer import NeuralLayer
from basic_nn import NeuralNetwork


logging = loggers.getLogger(__name__)


class LSTMLayer(NeuralLayer):

    def __init__(self, size, target_size=-1, activation='tanh', noise=0., dropouts=0., update_h0=True,
                 beta=0, optimization="SGD"):
        """
        Simple RNN Layer, input x sequence, output y sequence, cost, update parameters.
        Train a RNN without BPTT layers, which means the history_len should be set to 0 for the training data.
        BPTT is conducted once for every piece of data
        :return:
        """
        super(LSTMLayer, self).__init__(size, activation, noise, dropouts)
        self.learning_rate = 0.05
        self.weight_l2 = 0.0001
        self.update_h0 = update_h0
        self.disable_bias = True
        self.target_size = target_size
        self.optimization = optimization
        self.optimizer = MiniBatchOptimizer(batch_size=128, realtime=False)
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

    def _lstm_step(self, x_t, h_p, c_p):

        i_t_preact = T.dot(x_t, self.W_xi) + T.dot(h_p, self.W_hi) + T.dot(c_p, self.W_ci)# + self.B_i
        i_t = self._sigmoid(i_t_preact)
        f_t_preact = T.dot(x_t, self.W_xf) + T.dot(h_p, self.W_hf) + T.dot(c_p, self.W_cf)# + self.B_f
        f_t = self._sigmoid(f_t_preact)
        c_t_preact = T.dot(x_t, self.W_xc) + T.dot(h_p, self.W_hc)# + self.B_c
        tanh_c_t_right = self._tanh(c_t_preact)
        c_t = f_t * c_p + i_t * tanh_c_t_right
        o_t_preact = T.dot(x_t, self.W_xo) + T.dot(h_p, self.W_ho) + T.dot(c_t, self.W_co)# + self.B_o
        o_t = self._sigmoid(o_t_preact)
        tanh_c_t = self._tanh(c_t)
        h_t = o_t * tanh_c_t
        s_t_preact = T.dot(o_t, self.W_os)
        s_t = self._softmax(s_t_preact)

        return (s_t, h_t, c_t, o_t, f_t, i_t,
                i_t_preact, f_t_preact, c_t_preact, o_t_preact, s_t_preact, tanh_c_t, tanh_c_t_right, h_p, c_p)

    def _lstm_full_gradient_step(self, i,
                                 g_h, g_c,
                                 xs, ks, s_ts, h_ts, c_ts, o_ts, f_ts, i_ts,
                                 i_t_preacts, f_t_preacts, c_t_preacts, o_t_preacts, s_t_preacts,
                                 tanh_c_ts, tanh_c_t_rights, h_ps, c_ps):
        _s, = make_float_vectors("_s")
        _k = T.iscalar("_k")
        _softmax_s = self._softmax(_s)
        _g_cost = T.grad(self._cost_func(_softmax_s, _k), _s)
        g_s_t_preact = SRG(_g_cost, {_s: s_t_preacts[i], _softmax_s: s_ts[i], _k: ks[i]})

        g_wos = T.outer(o_ts[i], g_s_t_preact)
        g_o_t = T.dot(g_s_t_preact, T.transpose(self.W_os))# + 0.0002 * o_ts[i]
        g_o_t += tanh_c_ts[i] * g_h
        g_tanh_c_t = o_ts[i] * g_h
        g_c_t = self._tanh_grad(tanh_c_ts[i]) * g_tanh_c_t + g_c
        g_o_t_preact = self._sigmoid_grad(o_ts[i]) * g_o_t
        g_bo = g_o_t_preact
        g_h_p = T.dot(g_o_t_preact, T.transpose(self.W_ho))
        g_c_t += T.dot(g_o_t_preact, T.transpose(self.W_co))
        g_wxo = T.outer(xs[i], g_o_t_preact)
        g_who = T.outer(h_ps[i], g_o_t_preact)
        g_wco = T.outer(c_ts[i], g_o_t_preact)
        g_f_t = c_ps[i] * g_c_t
        g_c_p = f_ts[i] * g_c_t
        g_i_t = tanh_c_t_rights[i] * g_c_t
        g_tanh_c_t_right = i_ts[i] * g_c_t
        g_c_t_preact = self._tanh_grad(tanh_c_t_rights[i]) * g_tanh_c_t_right
        g_bc = g_c_t_preact
        g_wxc = T.outer(xs[i], g_c_t_preact)
        g_whc = T.outer(h_ps[i], g_c_t_preact)
        g_h_p += T.dot(g_c_t_preact, T.transpose(self.W_hc))
        g_f_t_preact = self._sigmoid_grad(f_ts[i]) * g_f_t
        g_wxf = T.outer(xs[i], g_f_t_preact)
        g_whf = T.outer(h_ps[i], g_f_t_preact)
        g_wcf = T.outer(c_ps[i], g_f_t_preact)
        g_bf = g_f_t_preact
        g_h_p += T.dot(g_f_t_preact, T.transpose(self.W_hf))
        g_c_p += T.dot(g_f_t_preact, T.transpose(self.W_cf))
        g_i_t_preact = self._sigmoid_grad(i_ts[i]) * g_i_t
        g_wxi = T.outer(xs[i], g_i_t_preact)
        g_whi = T.outer(h_ps[i], g_i_t_preact)
        g_wci = T.outer(c_ps[i], g_i_t_preact)
        g_bi = g_i_t_preact
        g_h_p += T.dot(g_i_t_preact, T.transpose(self.W_hi))
        g_c_p += T.dot(g_i_t_preact, T.transpose(self.W_ci))

        # Params:
        # self.W_xi,self.W_hi,self.W_ci,self.W_xf,self.W_hf,
        # self.W_cf,self.W_xc,self.W_hc,self.W_xo,self.W_ho,self.W_co,self.W_os
        return g_h_p, g_c_p, g_wxi, g_whi, g_wci, g_wxf, g_whf, g_wcf, g_wxc, g_whc, g_wxo, g_who, g_wco, g_wos# , \
               # g_bf, g_bi, g_bc, g_bo


    def _build_gradient_func(self):
        self._preact_t, = make_float_vectors("_preact")

        self._sigmoid_grad_act_t = self._sigmoid(self._preact_t)
        self._sigmoid_grad_t = T.grad(T.sum(self._sigmoid_grad_act_t), self._preact_t)
        self._sigmoid_grad = lambda act: SRG(self._sigmoid_grad_t, {self._sigmoid_grad_act_t: act})

        self._tanh_grad_act_t = self._tanh(self._preact_t)
        self._tanh_grad_t = T.grad(T.sum(self._tanh_grad_act_t), self._preact_t)
        self._tanh_grad = lambda act: SRG(self._tanh_grad_t, {self._tanh_grad_act_t: act})

        self._softmax_grad_act_t = self._softmax(self._preact_t)
        self._softmax_grad_t = T.grad(T.sum(self._softmax_grad_act_t), self._preact_t)
        self._softmax_grad = lambda preact, act: SRG(self._softmax_grad_t,
                                                     {self._softmax_grad_t: act, self._preact_t: preact})

    def _recurrent_func(self):

        # Run forward pass
        recurrent_vars, _ = theano.scan(fn=self._lstm_step, sequences=[self.x],
                                                outputs_info=[None, self.h0, self.c0] + [None]*12)
        # Full gradient back propagation
        self._build_gradient_func()
        gradient_vars, _ = theano.scan(fn=self._lstm_full_gradient_step,
                                       sequences=[T.arange(self.x.shape[0]-1, -1, -1)],
                                       outputs_info=[self.init_h, self.init_c]+[None for _ in self.params],
                                       non_sequences=[self.x, self._vars.k] + recurrent_vars)
        gradient_vars = gradient_vars[2:]
        gradients = [T.mean(g, axis=0) for g in gradient_vars]
        updates = optimize_parameters(self.params, gradients, lr=self.learning_rate, method=self.optimization,
                                      beta=self.beta, weight_l2=self.weight_l2, clip=True)

        return recurrent_vars[:3], updates

    def _predict_func(self):

        recurrent_vars, _ = theano.scan(fn=self._lstm_step, sequences=[self.x],
                                        outputs_info=[None, self.h0, self.c0] + [None]*12)
        s_list, h_list, c_list = recurrent_vars[:3]
        return s_list, [(self.h0, h_list[-1]), (self.c0, c_list[-1])]

    def _setup_functions(self):
        self._tanh = build_activation('tanh')
        self._sigmoid = build_activation('sigmoid')
        self._softmax = build_activation('softmax')
        [self.output_func, self.hidden_func, self.memory_func], recurrent_updates = self._recurrent_func()
        self.predict_func, self.predict_updates = self._predict_func()
        self.monitors.append(("last_h<0.1", 100 * (abs(self.hidden_func[-1]) < 0.1).mean()))
        self.monitors.append(("last_h<0.9", 100 * (abs(self.hidden_func[-1]) < 0.9).mean()))
        self.monitors.append(("c<0.1", 100 * (abs(self.memory_func[-1]) < 0.1).mean()))
        self.monitors.append(("c<0.9", 100 * (abs(self.memory_func[-1]) < 0.9).mean()))

        self.updates.extend(recurrent_updates)
        if self.update_h0:
            self.updates.append((self.h0, ifelse(T.eq(self._vars.k[-1], 0), self.init_h, self.hidden_func[-1])))
            self.updates.append((self.c0, ifelse(T.eq(self._vars.k[-1], 0), self.init_c, self.memory_func[-1])))


    def _setup_params(self):
        if self.target_size < 0:
            self.target_size = self.input_n

        self.h0 = theano.shared(value=np.zeros((self.output_n,), dtype=FLOATX), name='h_input')
        self.c0 = theano.shared(value=np.zeros((self.output_n,), dtype=FLOATX), name='c_input')

        self.init_h = theano.shared(value=self.h0.get_value(), name='init_h')
        self.init_c = theano.shared(value=self.c0.get_value(), name='init_c')
        self.zero_vector = theano.shared(value=np.zeros((self.output_n,), dtype=FLOATX), name='zero_h')

        self.W_xi = self.create_weight(self.input_n, self.output_n, "xi", scale=0.08)
        self.W_hi = self.create_weight(self.output_n, self.output_n, "hi", scale=0.08)
        self.W_ci = self.create_weight(self.output_n, self.output_n, "ci", scale=0.08)
        self.W_xf = self.create_weight(self.input_n, self.output_n, "xf", scale=0.08)
        self.W_hf = self.create_weight(self.output_n, self.output_n, "hf", scale=0.08)
        self.W_cf = self.create_weight(self.output_n, self.output_n, "cf", scale=0.08)
        self.W_xc = self.create_weight(self.input_n, self.output_n, "xc", scale=0.08)
        self.W_hc = self.create_weight(self.output_n, self.output_n, "hc", scale=0.08)
        self.W_xo = self.create_weight(self.input_n, self.output_n, "xo", scale=0.08)
        self.W_ho = self.create_weight(self.output_n, self.output_n, "ho", scale=0.08)
        self.W_co = self.create_weight(self.output_n, self.output_n, "co", scale=0.08)
        self.W_os = self.create_weight(self.output_n, self.target_size, "os", scale=0.08)

        self.B_f = self.create_bias(self.output_n, "f", value=0.7)
        self.B_i = self.create_bias(self.output_n, "i", value=0.)
        self.B_c = self.create_bias(self.output_n, "c", value=0.)
        self.B_o = self.create_bias(self.output_n, "o", value=0.)

        # Don't register parameters to the weights or bias
        # Update inside the recurrent steps
        self.W = []
        self.params = [self.W_xi,self.W_hi,self.W_ci,self.W_xf,self.W_hf,
                       self.W_cf,self.W_xc,self.W_hc,self.W_xo,self.W_ho,self.W_co,self.W_os]
                       #self.B_f, self.B_i, self.B_c, self.B_o]

    def clear_hidden(self):
        self.h0.set_value(np.zeros((self.output_n,), dtype=FLOATX))
        self.c0.set_value(np.zeros((self.output_n,), dtype=FLOATX))

    # def updating_callback(self):
    #     self.optimizer.run()


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

    # def iteration_callback(self):
    #     pass