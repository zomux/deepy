#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import NeuralLayer
from var import NeuralVariable
from deepy.utils import build_activation, FLOATX
import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

OUTPUT_TYPES = ["sequence", "one"]
INPUT_TYPES = ["sequence", "one"]

class LSTM(NeuralLayer):
    """
    Long short-term memory layer.
    """

    def __init__(self, hidden_size, input_type="sequence", output_type="sequence",
                 inner_activation="sigmoid", outer_activation="tanh",
                 inner_init=None, outer_init=None, steps=None,
                 go_backwards=False,
                 persistent_state=False, batch_size=0,
                 reset_state_for_input=None, forget_bias=1,
                 mask=None,
                 second_input=None, second_input_size=None):
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
        self.go_backwards = go_backwards
        # mask
        mask = mask.tensor if type(mask) == NeuralVariable else mask
        self.mask = mask.dimshuffle((1,0)) if mask else None
        self._sequence_map = OrderedDict()
        # second input
        if type(second_input) == NeuralVariable:
            second_input_size = second_input.dim()
            second_input = second_input.tensor

        self.second_input = second_input
        self.second_input_size = second_input_size
        self.forget_bias = forget_bias
        if input_type not in INPUT_TYPES:
            raise Exception("Input type of LSTM is wrong: %s" % input_type)
        if output_type not in OUTPUT_TYPES:
            raise Exception("Output type of LSTM is wrong: %s" % output_type)
        if self.persistent_state and not self.batch_size:
            raise Exception("Batch size must be set for persistent state mode")
        if mask and input_type == "one":
            raise Exception("Mask only works with sequence input")

    def _auto_reset_memories(self, x, h, m):
        reset_matrix = T.neq(x[:, self.reset_state_for_input], 1).dimshuffle(0, 'x')
        h = h * reset_matrix
        m = m * reset_matrix
        return h, m

    def step(self, *vars):
        # Parse sequence
        sequence_map = dict(zip(self._sequence_map.keys(), vars[:len(self._sequence_map)]))
        h_tm1, c_tm1 = vars[-2:]
        # Reset state
        if self.reset_state_for_input != None:
            h_tm1, c_tm1 = self._auto_reset_memories(sequence_map["x"], h_tm1, c_tm1)

        if self._input_type == "sequence":
            xi_t, xf_t, xo_t, xc_t = map(sequence_map.get, ["xi", "xf", "xo", "xc"])
        else:
            xi_t, xf_t, xo_t, xc_t = 0, 0, 0, 0

        # Add second input
        if "xi2" in sequence_map:
            xi2, xf2, xo2, xc2 = map(sequence_map.get, ["xi2", "xf2", "xo2", "xc2"])
            xi_t += xi2
            xf_t += xf2
            xo_t += xo2
            xc_t += xc2
        # LSTM core step
        i_t = self._inner_act(xi_t + T.dot(h_tm1, self.U_i) + self.b_i)
        f_t = self._inner_act(xf_t + T.dot(h_tm1, self.U_f) + self.b_f)
        c_t = f_t * c_tm1 + i_t * self._outer_act(xc_t + T.dot(h_tm1, self.U_c) + self.b_c)
        o_t = self._inner_act(xo_t + T.dot(h_tm1, self.U_o) + self.b_o)
        h_t = o_t * self._outer_act(c_t)
        # Apply mask
        if "mask" in sequence_map:
            mask = sequence_map["mask"].dimshuffle(0, 'x')
            h_t = h_t * mask + h_tm1 * (1 - mask)
            c_t = c_t * mask + c_tm1 * (1 - mask)
        return h_t, c_t

    def produce_input_sequences(self, x, mask=None, second_input=None):
        # Create sequence map
        self._sequence_map.clear()
        if self._input_type == "sequence":
            # Input vars
            xi = T.dot(x, self.W_i)
            xf = T.dot(x, self.W_f)
            xc = T.dot(x, self.W_c)
            xo = T.dot(x, self.W_o)
            self._sequence_map.update([("xi", xi), ("xf", xf), ("xc", xc), ("xo", xo)])
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
            xi2 = T.dot(second_input, self.W_i2)
            xf2 = T.dot(second_input, self.W_f2)
            xc2 = T.dot(second_input, self.W_c2)
            xo2 = T.dot(second_input, self.W_o2)
            self._sequence_map.update([("xi2", xi2), ("xf2", xf2), ("xc2", xc2), ("xo2", xo2)])
        return self._sequence_map.values()

    def produce_initial_states(self, x):
        if self.persistent_state:
            return self.state_h, self.state_m
        else:
            h0 = T.alloc(np.cast[FLOATX](0.), x.shape[0], self._hidden_size)
            m0 = h0
            return h0, m0

    def compute_tensor(self, x):
        h0, m0 = self.produce_initial_states(x)
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
            outputs_info=[h0, m0],
            n_steps=self._steps,
            go_backwards=self.go_backwards
        )

        # Save persistent state
        if self.persistent_state:
            self.register_updates((self.state_h, hiddens[-1]))
            self.register_updates((self.state_m, memories[-1]))

        if self._output_type == "one":
            return hiddens[-1]
        elif self._output_type == "sequence":
            return hiddens.dimshuffle((1,0,2))


    def prepare(self):
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
        if self.forget_bias > 0:
            self.b_f.set_value(np.ones((self._hidden_size,), dtype=FLOATX))

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
        # Second input
        if self.second_input_size:
            self.W_i2 = self.create_weight(self.second_input_size, self._hidden_size, "wi2", initializer=self._outer_init)
            self.W_f2 = self.create_weight(self.second_input_size, self._hidden_size, "wf2", initializer=self._outer_init)
            self.W_c2 = self.create_weight(self.second_input_size, self._hidden_size, "wc2", initializer=self._outer_init)
            self.W_o2 = self.create_weight(self.second_input_size, self._hidden_size, "wo2", initializer=self._outer_init)
            self.register_parameters(self.W_i2, self.W_f2, self.W_c2, self.W_o2)

        # Create persistent state
        if self.persistent_state:
            self.state_h = self.create_matrix(self.batch_size, self._hidden_size, "lstm_state_h")
            self.state_m = self.create_matrix(self.batch_size, self._hidden_size, "lstm_state_m")
            self.register_free_parameters(self.state_h, self.state_m)
        else:
            self.state_h = None
            self.state_m = None
