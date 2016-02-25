#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano.tensor as T
from recurrent import RecurrentLayer
from deepy.utils import neural_computation

OUTPUT_TYPES = ["sequence", "one"]
INPUT_TYPES = ["sequence", "one"]


class GRU(RecurrentLayer):
    def __init__(self, hidden_size, **kwargs):
        kwargs["hidden_size"] = hidden_size
        super(GRU, self).__init__("GRU", ["state"], **kwargs)

    @neural_computation
    def compute_new_state(self, step_inputs):
        xz_t, xr_t, xh_t, h_tm1 = map(step_inputs.get, ["xz_t", "xr_t", "xh_t", "state"])
        if not xz_t:
            xz_t, xr_t, xh_t = 0, 0, 0

        z_t = self.gate_activate(xz_t + T.dot(h_tm1, self.U_z) + self.b_z)
        r_t = self.gate_activate(xr_t + T.dot(h_tm1, self.U_r) + self.b_r)
        h_t_pre = self.activate(xh_t + T.dot(r_t * h_tm1, self.U_h) + self.b_h)
        h_t = z_t * h_tm1 + (1 - z_t) * h_t_pre

        return {"state": h_t}

    @neural_computation
    def merge_inputs(self, input_var, additional_inputs=None):
        if not additional_inputs:
            additional_inputs = []
        all_inputs = filter(bool, [input_var] + additional_inputs)
        z_inputs = []
        r_inputs = []
        h_inputs = []
        for x, weights in zip(all_inputs, self.input_weights):
            wz, wr, wh = weights
            z_inputs.append(T.dot(x, wz))
            r_inputs.append(T.dot(x, wr))
            h_inputs.append(T.dot(x, wh))
        merged_inputs = {
            "xz_t": sum(z_inputs),
            "xr_t": sum(r_inputs),
            "xh_t": sum(h_inputs)
        }
        return merged_inputs

    def prepare(self):
        self.output_dim = self.hidden_size

        self.U_z = self.create_weight(self.hidden_size, self.hidden_size, "uz", initializer=self.inner_init)
        self.b_z = self.create_bias(self.hidden_size, "z")

        self.U_r = self.create_weight(self.hidden_size, self.hidden_size, "ur", initializer=self.inner_init)
        self.b_r = self.create_bias(self.hidden_size, "r")

        self.U_h = self.create_weight(self.hidden_size, self.hidden_size, "uh", initializer=self.inner_init)
        self.b_h = self.create_bias(self.hidden_size, "h")

        self.register_parameters(self.U_z, self.b_z,
                                 self.U_r, self.b_r,
                                 self.U_h, self.b_h)

        self.input_weights = []
        if self._input_type == "sequence":
            all_input_dims = [self.input_dim] + self.additional_input_dims
        else:
            all_input_dims = self.additional_input_dims
        for i, input_dim in enumerate(all_input_dims):
            wz = self.create_weight(input_dim, self.hidden_size, "wz_{}".format(i + 1), initializer=self.outer_init)
            wr = self.create_weight(input_dim, self.hidden_size, "wr_{}".format(i + 1), initializer=self.outer_init)
            wh = self.create_weight(input_dim, self.hidden_size, "wh_{}".format(i + 1), initializer=self.outer_init)
            weights = [wz, wr, wh]
            self.input_weights.append(weights)
            self.register_parameters(*weights)
