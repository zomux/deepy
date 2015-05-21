#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import NeuralLayer
from deepy.utils import build_activation, FLOATX
import numpy as np
import theano
import theano.tensor as T

OUTPUT_TYPES = ["sequence", "one"]
INPUT_TYPES = ["sequence", "one"]

class LSTM(NeuralLayer):
    """
    Long short-term memory layer.
    """

    def __init__(self, hidden_size, output_size=None, input_type="sequence", output_type="sequence",
                 inner_activation="tanh", outer_activation="tanh",
                 inner_init=None, outer_init=None, steps=None):
        super(LSTM, self).__init__("lstm")
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._input_type = input_type
        self._output_type = output_type
        self._inner_activation = inner_activation
        self._outer_activation = outer_activation
        self._inner_init = inner_init
        self._outer_init = outer_init
        self._steps = steps
        if input_type not in INPUT_TYPES:
            raise Exception("Input type of LSTM is wrong: %s" % input_type)
        if output_type not in OUTPUT_TYPES:
            raise Exception("Output type of LSTM is wrong: %s" % output_type)

    def step(self, *vars):
        if self._input_type == "sequence":
            xi_t, xf_t, xo_t, xc_t, h_tm1, c_tm1 = vars
            i_t = self._inner_act(xi_t + T.dot(h_tm1, self.U_i))
            f_t = self._inner_act(xf_t + T.dot(h_tm1, self.U_f))
            c_t = f_t * c_tm1 + i_t * self._outer_act(xc_t + T.dot(h_tm1, self.U_c))
            o_t = self._inner_act(xo_t + T.dot(h_tm1, self.U_o))
            h_t = o_t * self._outer_act(c_t)
        else:
            h_tm1, c_tm1 = vars
            i_t = self._inner_act(T.dot(h_tm1, self.U_i) + self.b_i)
            f_t = self._inner_act(T.dot(h_tm1, self.U_f) + self.b_f)
            c_t = f_t * c_tm1 + i_t * self._outer_act(T.dot(h_tm1, self.U_c) + self.b_c)
            o_t = self._inner_act(T.dot(h_tm1, self.U_o) + self.b_o)
            h_t = o_t * self._outer_act(c_t)

        return h_t, c_t

    def produce_input_sequences(self, x):
        xi = T.dot(x, self.W_i) + self.b_i
        xf = T.dot(x, self.W_f) + self.b_f
        xc = T.dot(x, self.W_c) + self.b_c
        xo = T.dot(x, self.W_o) + self.b_o
        sequences = [xi, xf, xo, xc]
        return sequences

    def produce_initial_states(self, x):
        h0 = T.alloc(np.cast[FLOATX](0.), x.shape[0], self._hidden_size)
        m0 = h0
        return h0, m0

    def output(self, x):
        h0, m0 = self.produce_initial_states(x)
        if self._input_type == "sequence":
            # Move middle dimension to left-most position
            # (sequence, batch, value)
            x = x.dimshuffle((1,0,2))
            sequences = self.produce_input_sequences(x)
        else:
            h0 = x
            sequences = []

        [hiddens, memories], _ = theano.scan(
            self.step,
            sequences=sequences,
            outputs_info=[h0, m0],
            # non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c]
        )

        if self._output_type == "one":
            return hiddens[-1]
        elif self._output_type == "sequence":
            return hiddens.dimshuffle((1,0,2))


    def setup(self):
        self._setup_params()
        self._setup_functions()

    def _setup_functions(self):
        self._inner_act = build_activation(self._inner_activation)
        self._outer_act = build_activation(self._outer_activation)

    def _setup_params(self):
        self.output_dim = self._hidden_size

        self.W_i = self.create_weight(self.input_dim, self._hidden_size, "wi", initializer=self._outer_init)
        self.U_i = self.create_weight(self._hidden_size, self._hidden_size, "ui", initializer=self._inner_init)
        self.b_i = self.create_bias(self._hidden_size, "i")

        self.W_f = self.create_weight(self.input_dim, self._hidden_size, "wf", initializer=self._outer_init)
        self.U_f = self.create_weight(self._hidden_size, self._hidden_size, "uf", initializer=self._inner_init)
        self.b_f = self.create_bias(self._hidden_size, "f")

        self.W_c = self.create_weight(self.input_dim, self._hidden_size, "wc", initializer=self._outer_init)
        self.U_c = self.create_weight(self._hidden_size, self._hidden_size, "uc", initializer=self._inner_init)
        self.b_c = self.create_bias(self._hidden_size, "c")

        self.W_o = self.create_weight(self.input_dim, self._hidden_size, "wo", initializer=self._outer_init)
        self.U_o = self.create_weight(self._hidden_size, self._hidden_size, "uo", initializer=self._inner_init)
        self.b_o = self.create_bias(self._hidden_size, suffix="o")


        if self._input_type == "sequence":
            self.register_parameters(self.W_i, self.U_i, self.b_i,
                                     self.W_c, self.U_c, self.b_c,
                                     self.W_f, self.U_f, self.b_f,
                                     self.W_o, self.U_o, self.b_o)
        else:
            self.register_parameters(self.U_i, self.b_i,
                                     self.U_c, self.b_c,
                                     self.U_f, self.b_f,
                                     self.U_o, self.b_o)