#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import NeuralLayer
from deepy.utils import build_activation, FLOATX
import numpy as np
import theano
import theano.tensor as T

OUTPUT_TYPES = ["sequence", "one"]
INPUT_TYPES = ["sequence", "one"]

class RNN(NeuralLayer):
    """
    Recurrent neural network layer.
    """

    def __init__(self, hidden_size, input_type="sequence", output_type="sequence", vector_core=None,
                 hidden_activation="tanh", hidden_init=None, input_init=None, steps=None,
                 persistent_state=False, reset_state_for_input=None, batch_size=None):
        super(RNN, self).__init__("rnn")
        self._hidden_size = hidden_size
        self.output_dim = self._hidden_size
        self._input_type = input_type
        self._output_type = output_type
        self._hidden_activation = hidden_activation
        self._hidden_init = hidden_init
        self._vector_core = vector_core
        self._input_init = input_init
        self.persistent_state = persistent_state
        self.reset_state_for_input = reset_state_for_input
        self.batch_size = batch_size
        self._steps = steps
        if input_type not in INPUT_TYPES:
            raise Exception("Input type of RNN is wrong: %s" % input_type)
        if output_type not in OUTPUT_TYPES:
            raise Exception("Output type of RNN is wrong: %s" % output_type)
        if self.persistent_state and not self.batch_size:
            raise Exception("Batch size must be set for persistent state mode")

    def _hidden_preact(self, h):
        return T.dot(h, self.W_h) if not self._vector_core else h * self.W_h


    def step(self, *variables):
        if self._input_type == "sequence":
            x, h = variables
            # Reset part of the state on condition
            if self.reset_state_for_input != None:
                h = h * T.neq(x[:, self.reset_state_for_input], 1).dimshuffle(0, 'x')
            z = T.dot(x, self.W_i) + self._hidden_preact(h) + self.B_h
        else:
            h, = variables
            z = self._hidden_preact(h) + self.B_h

        new_h = self._hidden_act(z)
        return new_h


    def output(self, x):
        sequences = []
        h0 = T.alloc(np.cast[FLOATX](0.), x.shape[0], self._hidden_size)
        if self._input_type == "sequence":
            # Move middle dimension to left-most position
            # (sequence, batch, value)
            sequences = [x.dimshuffle((1,0,2))]
            # Set initial state
            if self.persistent_state:
                h0 = self.state
        else:
            h0 = x
        step_outputs = [h0]
        hiddens, _ = theano.scan(self.step, sequences=sequences, outputs_info=step_outputs, n_steps=self._steps)

        # Save persistent state
        if self.persistent_state:
            self.register_updates((self.state, hiddens[-1]))

        if self._output_type == "one":
            return hiddens[-1]
        elif self._output_type == "sequence":
            return hiddens.dimshuffle((1,0,2))

    def setup(self):
        print self.input_dim, self._hidden_size
        if self._input_type == "one" and self.input_dim != self._hidden_size:
            raise Exception("For RNN receives one vector as input, "
                            "the hidden size should be same as last output dimension.")
        self._setup_params()
        self._setup_functions()

    def _setup_functions(self):
        self._hidden_act = build_activation(self._hidden_activation)

    def _setup_params(self):
        if not self._vector_core:
            self.W_h = self.create_weight(self._hidden_size, self._hidden_size, suffix="h", initializer=self._hidden_init)
        else:
            self.W_h = self.create_bias(self._hidden_size, suffix="h")
            self.W_h.set_value(self.W_h.get_value() + self._vector_core)
        self.B_h = self.create_bias(self._hidden_size, suffix="h")

        self.register_parameters(self.W_h, self.B_h)

        if self.persistent_state:
            self.state = self.create_matrix(self.batch_size, self._hidden_size, "rnn_state")
            self.register_free_parameters(self.state)
        else:
            self.state = None

        if self._input_type == "sequence":
            self.W_i = self.create_weight(self.input_dim, self._hidden_size, suffix="i", initializer=self._input_init)
            self.register_parameters(self.W_i)


