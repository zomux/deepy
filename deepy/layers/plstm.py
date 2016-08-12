#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano.tensor as T
from recurrent import RecurrentLayer
from deepy.utils import neural_computation, FLOATX

class PeepholeLSTM(RecurrentLayer):
    """
    Long short-term memory layer with peepholes.
    """

    def __init__(self, hidden_size, init_forget_bias=1, **kwargs):
        kwargs["hidden_size"] = hidden_size
        super(PeepholeLSTM, self).__init__("PLSTM", ["state", "lstm_cell"], **kwargs)
        self._init_forget_bias = 1

    @neural_computation
    def compute_new_state(self, step_inputs):
        xi_t, xf_t, xo_t, xc_t, h_tm1, c_tm1 = map(step_inputs.get, ["xi", "xf", "xc", "xo", "state", "lstm_cell"])
        if not xi_t:
            xi_t, xf_t, xo_t, xc_t = 0, 0, 0, 0

        # LSTM core step
        hs = self.hidden_size
        dot_h = T.dot(h_tm1, self.U)
        dot_c = T.dot(h_tm1, self.C)
        i_t = self.gate_activate(xi_t + dot_h[:, :hs] + self.b_i + dot_c[:, :hs])
        f_t = self.gate_activate(xf_t + dot_h[:, hs:hs*2] + self.b_f + dot_c[:, hs:hs*2])
        c_t = f_t * c_tm1 + i_t * self.activate(xc_t + dot_h[:, hs*2:hs*3] + self.b_c)
        o_t = self.gate_activate(xo_t + dot_h[:, hs*3:hs*4] + dot_c[:, hs*2:hs*3] + self.b_o)
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
        dot_input = T.dot(merged_input, self.W)
        merged_inputs = {
            "xi": dot_input[:, :, :self.hidden_size],
            "xf": dot_input[:, :, self.hidden_size:self.hidden_size*2],
            "xc": dot_input[:, :, self.hidden_size*2:self.hidden_size*3],
            "xo": dot_input[:, :, self.hidden_size*3:self.hidden_size*4],
        }
        return merged_inputs


    def prepare(self):
        if self._input_type == "sequence":
            all_input_dims = [self.input_dim] + self.additional_input_dims
        else:
            all_input_dims = self.additional_input_dims
        summed_input_dim = sum(all_input_dims, 0)
        self.output_dim = self.hidden_size

        self.W = self.create_weight(summed_input_dim, self.hidden_size * 4, "W", initializer=self.outer_init)
        self.U = self.create_weight(self.hidden_size, self.hidden_size * 4, "U", initializer=self.inner_init)
        self.C = self.create_weight(self.hidden_size, self.hidden_size * 3, "C", initializer=self.inner_init)

        self.b_i = self.create_bias(self.hidden_size, "bi")
        self.b_f = self.create_bias(self.hidden_size, "bf")
        self.b_f.set_value(np.ones((self.hidden_size,) * self._init_forget_bias, dtype=FLOATX))
        self.b_c = self.create_bias(self.hidden_size, "bc")
        self.b_o = self.create_bias(self.hidden_size, "bo")


        if summed_input_dim > 0:
            self.register_parameters(self.W, self.U, self.C,
                                     self.b_i, self.b_f, self.b_c, self.b_o)
        else:
            self.register_parameters(self.U, self.C,
                                     self.b_i, self.b_f, self.b_c, self.b_o)