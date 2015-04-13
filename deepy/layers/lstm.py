#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import NeuralLayer
from deepy.util import build_activation, FLOATX
import theano
import theano.tensor as T
import numpy as np

class LSTM(NeuralLayer):
    """
    Long short-term memory layer.
    """
    def __init__(self, output_size=128,
        activation='tanh', inner_activation='tanh',
        weights=None, truncate_gradient=-1, return_sequences=False):
        super(LSTM, self).__init__("LSTM")

        self.output_dim = output_size
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.activation = build_activation(activation)
        self.inner_activation = build_activation(inner_activation)

    def setup(self):
        self.W_i = self.create_weight(self.input_dim, self.output_dim, "i")
        self.U_i = self.create_weight(self.output_dim, self.output_dim, "ui")
        self.b_i = self.create_bias(self.output_dim, "i")

        self.W_f = self.create_weight(self.input_dim, self.output_dim)
        self.U_f = self.create_weight(self.output_dim, self.output_dim)
        self.b_f = self.create_bias(self.output_dim)

        self.W_c = self.create_weight(self.input_dim, self.output_dim)
        self.U_c = self.create_weight(self.output_dim, self.output_dim)
        self.b_c = self.create_bias(self.output_dim)

        self.W_o = self.create_weight(self.input_dim, self.output_dim)
        self.U_o = self.create_weight(self.output_dim, self.output_dim)
        self.b_o = self.create_bias(self.output_dim, suffix="o")

        self.register_parameters(self.W_i, self.U_i, self.b_i,
                                 self.W_c, self.U_c, self.b_c,
                                 self.W_f, self.U_f, self.b_f,
                                 self.W_o, self.U_o, self.b_o)

    def _step(self,
        xi_t, xf_t, xo_t, xc_t,
        h_tm1, c_tm1,
        u_i, u_f, u_o, u_c):
        i_t = self.inner_activation(xi_t + T.dot(h_tm1, u_i))
        f_t = self.inner_activation(xf_t + T.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1, u_c))
        o_t = self.inner_activation(xo_t + T.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def output(self, X):
        X = X.dimshuffle((1,0,2))

        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o

        [outputs, memories], updates = theano.scan(
            self._step,
            sequences=[xi, xf, xo, xc],
            outputs_info=[
                T.alloc(np.cast[FLOATX](0.), X.shape[1], self.output_dim),
                T.alloc(np.cast[FLOATX](0.), X.shape[1], self.output_dim)
            ],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]