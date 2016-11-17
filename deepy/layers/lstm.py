#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano.tensor as T
from recurrent import RecurrentLayer
from deepy.core.decorations import neural_computation
from deepy.core.global_env import env


class LSTM(RecurrentLayer):
    """
    Long short-term memory layer.
    """

    def __init__(self, hidden_size, init_forget_bias=1, **kwargs):
        kwargs["hidden_size"] = hidden_size
        super(LSTM, self).__init__("LSTM", ["state", "lstm_cell"], **kwargs)
        self._init_forget_bias = 1

    @neural_computation
    def compute_new_state(self, step_inputs):
        xi_t, xf_t, xo_t, xc_t, h_tm1, c_tm1 = map(step_inputs.get, ["xi", "xf", "xc", "xo", "state", "lstm_cell"])
        if not xi_t:
            xi_t, xf_t, xo_t, xc_t = 0, 0, 0, 0

        # LSTM core step
        i_t = self.gate_activate(xi_t + T.dot(h_tm1, self.U_i) + self.b_i)
        f_t = self.gate_activate(xf_t + T.dot(h_tm1, self.U_f) + self.b_f)
        c_t = f_t * c_tm1 + i_t * self.activate(xc_t + T.dot(h_tm1, self.U_c) + self.b_c)
        o_t = self.gate_activate(xo_t + T.dot(h_tm1, self.U_o) + self.b_o)
        h_t = o_t * self.activate(c_t)

        return {"state": h_t, "lstm_cell": c_t}

    @neural_computation
    def merge_inputs(self, input_var, additional_inputs=None):
        if not additional_inputs:
            additional_inputs = []
        all_inputs = filter(bool, [input_var] + additional_inputs)
        if not all_inputs:
            return {}
        last_dim_id = all_inputs[0].ndim - 1
        merged_input = T.concatenate(all_inputs, axis=last_dim_id)
        merged_inputs = {
            "xi": T.dot(merged_input, self.W_i),
            "xf": T.dot(merged_input, self.W_f),
            "xc": T.dot(merged_input, self.W_c),
            "xo": T.dot(merged_input, self.W_o),
        }
        return merged_inputs


    def prepare(self):
        if self._input_type == "sequence":
            all_input_dims = [self.input_dim] + self.additional_input_dims
        else:
            all_input_dims = self.additional_input_dims
        summed_input_dim = sum(all_input_dims, 0)
        self.output_dim = self.hidden_size

        self.W_i = self.create_weight(summed_input_dim, self.hidden_size, "wi", initializer=self.outer_init)
        self.U_i = self.create_weight(self.hidden_size, self.hidden_size, "ui", initializer=self.inner_init)
        self.b_i = self.create_bias(self.hidden_size, "i")

        self.W_f = self.create_weight(summed_input_dim, self.hidden_size, "wf", initializer=self.outer_init)
        self.U_f = self.create_weight(self.hidden_size, self.hidden_size, "uf", initializer=self.inner_init)
        self.b_f = self.create_bias(self.hidden_size, "f")
        self.b_f.set_value(np.ones((self.hidden_size,) * self._init_forget_bias, dtype=env.FLOATX))

        self.W_c = self.create_weight(summed_input_dim, self.hidden_size, "wc", initializer=self.outer_init)
        self.U_c = self.create_weight(self.hidden_size, self.hidden_size, "uc", initializer=self.inner_init)
        self.b_c = self.create_bias(self.hidden_size, "c")

        self.W_o = self.create_weight(summed_input_dim, self.hidden_size, "wo", initializer=self.outer_init)
        self.U_o = self.create_weight(self.hidden_size, self.hidden_size, "uo", initializer=self.inner_init)
        self.b_o = self.create_bias(self.hidden_size, suffix="o")


        if summed_input_dim > 0:
            self.register_parameters(self.W_i, self.U_i, self.b_i,
                                     self.W_c, self.U_c, self.b_c,
                                     self.W_f, self.U_f, self.b_f,
                                     self.W_o, self.U_o, self.b_o)
        else:
            self.register_parameters(self.U_i, self.b_i,
                                     self.U_c, self.b_c,
                                     self.U_f, self.b_f,
                                     self.U_o, self.b_o)

