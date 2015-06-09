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

    def __init__(self, hidden_size, input_type="sequence", output_type="sequence",
                 inner_activation="sigmoid", outer_activation="tanh",
                 inner_init=None, outer_init=None, steps=None,
                 persistent_state=False, batch_size=0,
                 reset_state_for_input=None):
        super(LSTM, self).__init__("lstm")
        self._hidden_size = hidden_size
        self._input_type = input_type
        self._output_type = output_type
        self._inner_activation = inner_activation
        self._outer_activation = outer_activation
        self._inner_init = inner_init
        self._outer_init = outer_init
        self._steps = steps
        self.persistent_state = persistent_state
        self.reset_state_for_input = reset_state_for_input
        self.batch_size = batch_size
        if input_type not in INPUT_TYPES:
            raise Exception("Input type of LSTM is wrong: %s" % input_type)
        if output_type not in OUTPUT_TYPES:
            raise Exception("Output type of LSTM is wrong: %s" % output_type)
        if self.persistent_state and not self.batch_size:
            raise Exception("Batch size must be set for persistent state mode")

    def _auto_reset_memories(self, x, h, m):
        reset_matrix = T.neq(x[:, self.reset_state_for_input], 1).dimshuffle(0, 'x')
        h = h * reset_matrix
        m = m * reset_matrix
        return h, m

    def _parse_sequential_vars(self, vars):
        # Get raw input
        if self.reset_state_for_input != None:
            x = vars[0]
            vars = vars[1:]
        # Unpack varables
        xi_t, xf_t, xo_t, xc_t, h_tm1, c_tm1 = vars
        # Auto reset states
        if self.reset_state_for_input != None:
            h_tm1, c_tm1 = self._auto_reset_memories(x, h_tm1, c_tm1)
        return xi_t, xf_t, xo_t, xc_t, h_tm1, c_tm1

    def step(self, *vars):
        if self._input_type == "sequence":
            xi_t, xf_t, xo_t, xc_t, h_tm1, c_tm1 = self._parse_sequential_vars(vars)
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
        if self.reset_state_for_input != None:
            sequences.insert(0, x)
        return sequences

    def produce_initial_states(self, x):
        if self.persistent_state:
            return self.state_h, self.state_m
        else:
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

        # Save persistent state
        if self.persistent_state:
            self.register_updates((self.state_h, hiddens[-1]))
            self.register_updates((self.state_m, memories[-1]))

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
        # Create persistent state
        if self.persistent_state:
            self.state_h = self.create_matrix(self.batch_size, self._hidden_size, "lstm_state_h")
            self.state_m = self.create_matrix(self.batch_size, self._hidden_size, "lstm_state_m")
            self.register_free_parameters(self.state_h, self.state_m)
        else:
            self.state_h = None
            self.state_m = None