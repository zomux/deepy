#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import NeuralLayer
from deepy.utils import build_activation, FLOATX
import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

OUTPUT_TYPES = ["sequence", "one"]
INPUT_TYPES = ["sequence", "one"]

class GRU(NeuralLayer):
    """
    Gated recurrent unit.
    """

    def __init__(self, hidden_size, input_type="sequence", output_type="sequence",
                 inner_activation="hard_sigmoid", outer_activation="sigmoid",
                 inner_init="orthogonal", outer_init=None, steps=None,
                 go_backwards=False,
                 persistent_state=False, batch_size=0,
                 reset_state_for_input=None,
                 mask=None,
                 second_input=None, second_input_size=None):
        super(GRU, self).__init__("GRU")
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
        self.go_backwards = go_backwards
        self.mask = mask.dimshuffle((1,0)) if mask else None
        self._sequence_map = OrderedDict()
        self.second_input = second_input
        self.second_input_size = second_input_size
        if input_type not in INPUT_TYPES:
            raise Exception("Input type of GRU is wrong: %s" % input_type)
        if output_type not in OUTPUT_TYPES:
            raise Exception("Output type of GRU is wrong: %s" % output_type)
        if self.persistent_state and not self.batch_size:
            raise Exception("Batch size must be set for persistent state mode")
        if mask and input_type == "one":
            raise Exception("Mask only works with sequence input")

    def _auto_reset_memories(self, x, h):
        reset_matrix = T.neq(x[:, self.reset_state_for_input], 1).dimshuffle(0, 'x')
        h = h * reset_matrix
        return h

    def step(self, *vars):
        # Parse sequence
        sequence_map = dict(zip(self._sequence_map.keys(), vars[:len(self._sequence_map)]))
        h_tm1 = vars[-1:]
        # Reset state
        if self.reset_state_for_input != None:
            h_tm1 = self._auto_reset_memories(sequence_map["x"], h_tm1)

        if self._input_type == "sequence":
            xz_t, xr_t, xh_t = map(sequence_map.get, ["xz", "xr", "xh"])
        else:
            xz_t, xr_t, xh_t = 0, 0, 0

        # Add second input
        if "xi2" in sequence_map:
            xi2, xf2, xo2, xc2 = map(sequence_map.get, ["xz2", "xr2", "xh2"])
            xz_t += xi2
            xr_t += xf2
            xh_t += xc2
        # GRU core step
        z_t = self._inner_act(xz_t + T.dot(h_tm1, self.U_z))
        r_t = self._inner_act(xr_t + T.dot(h_tm1, self.U_r))
        h_t_pre = self._outer_act(xh_t + T.dot(r_t * h_tm1, self.U_h))
        h_t = z * h_tm1 + (1 - z) *  h_t_pre
        # Apply mask
        if "mask" in sequence_map:
            mask = sequence_map["mask"].dimshuffle(0, 'x')
            h_t = h_t * mask + h_tm1 * (1 - mask)
        return h_t

    def produce_input_sequences(self, x, mask=None, second_input=None):
        # Create sequence map
        self._sequence_map.clear()
        if self._input_type == "sequence":
            # Input vars
            xz = T.dot(x, self.W_i)
            xr = T.dot(x, self.W_f)
            xh = T.dot(x, self.W_c)
            self._sequence_map.update([("xz", xz), ("xr", xr), ("xh", xh)])
        # Reset state
        if self.reset_state_for_input != None:
            self._sequence_map["x"] = x
        # Add mask
        if mask:
            self._sequence_map["mask"] = mask
        elif self.mask:
            self._sequence_map["mask"] = self.mask
        # Add second input
        if self.second_input and not second_input:
            second_input = self.second_input
        if second_input:
            xz2 = T.dot(second_input, self.W_z2)
            xr2 = T.dot(second_input, self.W_r2)
            xh2 = T.dot(second_input, self.W_h2)
            self._sequence_map.update([("xz2", xz2), ("xr2", xr2), ("xh2", xh2)])
        return self._sequence_map.values()

    def produce_initial_state(self, x):
        if self.persistent_state:
            return self.state_h
        else:
            h0 = T.alloc(np.cast[FLOATX](0.), x.shape[0], self._hidden_size)
            return h0

    def output(self, x):
        h0 = self.produce_initial_state(x)
        if self._input_type == "sequence":
            # Move middle dimension to left-most position
            # (sequence, batch, value)
            x = x.dimshuffle((1,0,2))
            sequences = self.produce_input_sequences(x)
        else:
            h0 = x
            sequences = self.produce_input_sequences(None)

        [hiddens, memories], _ = theano.scan(
            self.step,
            sequences=sequences,
            outputs_info=[h0],
            n_steps=self._steps,
            go_backwards=self.go_backwards
        )

        # Save persistent state
        if self.persistent_state:
            self.register_updates((self.state_h, hiddens[-1]))

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

        self.W_z = self.create_weight(self.input_dim, self._hidden_size, "wz", initializer=self._outer_init)
        self.U_z = self.create_weight(self._hidden_size, self._hidden_size, "uz", initializer=self._inner_init)
        self.b_z = self.create_bias(self._hidden_size, "z")

        self.W_r = self.create_weight(self.input_dim, self._hidden_size, "wr", initializer=self._outer_init)
        self.U_r = self.create_weight(self._hidden_size, self._hidden_size, "ur", initializer=self._inner_init)
        self.b_r = self.create_bias(self._hidden_size, "r")

        self.W_h = self.create_weight(self.input_dim, self._hidden_size, "wh", initializer=self._outer_init)
        self.U_h = self.create_weight(self._hidden_size, self._hidden_size, "uh", initializer=self._inner_init)
        self.b_h = self.create_bias(self._hidden_size, "h")

        if self._input_type == "sequence":
            self.register_parameters(self.W_z, self.U_z, self.b_z,
                                     self.W_r, self.U_r, self.b_r
                                     self.W_h, self.U_h, self.b_h)
        else:
            self.register_parameters(self.U_z, self.b_z,
                                     self.U_r, self.b_r,
                                     self.U_h, self.b_h)
        # Second input
        if self.second_input_size:
            self.W_z2 = self.create_weight(self.second_input_size, self._hidden_size, "wz2", initializer=self._outer_init)
            self.W_r2 = self.create_weight(self.second_input_size, self._hidden_size, "wr2", initializer=self._outer_init)
            self.W_h2 = self.create_weight(self.second_input_size, self._hidden_size, "wh2", initializer=self._outer_init)
            self.register_parameters(self.W_i2, self.W_f2, self.W_c2)

        # Create persistent state
        if self.persistent_state:
            self.state_h = self.create_matrix(self.batch_size, self._hidden_size, "GRU_state_h")
            self.register_free_parameters(self.state_h)
        else:
            self.state_h = None
