#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging as loggers

import theano
import theano.tensor as T

from deepy.utils import build_activation
from deepy.trainers import THEANO_LINKER
from deepy.layers.layer import NeuralLayer

logging = loggers.getLogger(__name__)

from network import NeuralNetwork

# TODO: repair additional_h mode
class RecursiveAutoEncoder(NeuralNetwork):
    """
    Recursive auto encoder (Recursively encode a sequence by combining two children).
    Parameters:
        rep_dim - dimension of representation
    """
    def __init__(self, input_dim, rep_dim=None, activation='tanh', unfolding=True, additional_h=False,
                 config=None):
        super(RecursiveAutoEncoder, self).__init__(input_dim, config=config, input_tensor=3)

        self.rep_dim = rep_dim
        self.stack(RecursiveAutoEncoderCore(rep_dim, unfolding=unfolding, additional_h=additional_h))
        self._encode_func = None
        self._decode_func = None

    def _cost_func(self, y):
        # As the core returns cost
        return y

    @property
    def cost(self):
        return self._cost_func(self.output)

    @property
    def test_cost(self):
        return self._cost_func(self.test_output)

    def encode(self, x):
        """
        Encode given input.
        """
        if not self._encode_func:
            x_var = T.vector()
            self._encode_func = theano.function([x_var], self.layers[0].encode_func(x),
                            allow_input_downcast=True, mode=theano.Mode(linker=THEANO_LINKER))
        return self._encode_func(x)

    def decode(self, rep, n_steps):
        """
        Decode given representation.
        """
        if not self._decode_func:
            rep_var = T.vector()
            n_var = T.iscalar()
            self._decode_func = theano.function([rep_var, n_var], self.layers[0].decode_func(rep_var, n_var),
                            allow_input_downcast=True, mode=theano.Mode(linker=THEANO_LINKER))
        return self._decode_func(rep, n_steps)

class RecursiveAutoEncoderCore(NeuralLayer):

    def __init__(self, rep_dim=None, activation='tanh', unfolding=True, additional_h=True):
        """
        Binarized Recursive Encoder Core layer
        Input:
        A sequence of terminal nodes in vectore representations.
        Output:
        Cost
        """
        super(RecursiveAutoEncoderCore, self).__init__("RAE")
        self.rep_dim = rep_dim
        self.unfolding = unfolding
        self.additional_h = additional_h
        self.activation = activation

    def setup(self):
        self._setup_params()
        self._setup_functions()

    def output(self, x):
        rep, cost = self._recursive_func(x)
        self.register_monitors(("mean(rep)", abs(rep).mean()))
        return cost

    def _recursive_step(self, i, p, x):
        x_t = x[i]
        # Encoding
        rep = self._activation_func(T.dot(p, self.W_e1) + T.dot(x_t, self.W_e2) + self.B_e)
        if self.unfolding:
            x_decs = self._unfold(rep, i)
            distance = T.sum((x_decs - x[: i + 1]) ** 2)
        else:
            # Decoding
            p_dec, x_dec = self._decode_step(rep)
            # Euclidean distance
            distance = T.sum((p_dec - p)**2 + (x_dec - x_t)**2)
        return rep, distance

    def _unfold(self, p, n):
        if self.additional_h:
            n += 1
        [ps, xs], _ = theano.scan(self._decode_step, outputs_info=[p, None], n_steps=n)
        if self.additional_h:
            return xs[::-1]
        else:
            return T.concatenate([xs, [ps[-1]]])[::-1]

    def _recursive_func(self, x):
        # Return total error
        if self.additional_h:
            h0 = self.h0
            start_index = 0
        else:
            h0 = x[0]
            start_index = 1
        [reps, distances], _ = theano.scan(self._recursive_step, sequences=[T.arange(start_index, x.shape[0])],
                                           outputs_info=[h0, None], non_sequences=[x])
        return reps[-1], T.sum(distances)

    def encode_func(self, x):
        if self.additional_h:
            h0 = self.h0
            start_index = 0
        else:
            h0 = x[0]
            start_index = 1
        [reps, _], _ = theano.scan(self._recursive_step, sequences=[T.arange(start_index, x.shape[0])],
                                           outputs_info=[h0, None], non_sequences=[x])
        return reps[-1]

    def _decode_step(self, p):
        p_dec = self._activation_func(T.dot(p, self.W_d1) + self.B_d1)
        x_dec = self._activation_func(T.dot(p, self.W_d2) + self.B_d2)
        return  p_dec, x_dec

    def decode_func(self, rep, n):
        return self._unfold(rep, n)

    def _setup_functions(self):
        self._assistive_params = []
        self._activation_func = build_activation(self.activation)
        self._softmax_func = build_activation('softmax')

    def _setup_params(self):
        if not self.rep_dim or self.rep_dim < 0:
            self.rep_dim = self.input_dim
        if not self.additional_h and self.rep_dim != self.input_dim:
            raise Exception("rep_dim must be input_dim when additional_h is not used")

        self.W_e1 = self.create_weight(self.rep_dim, self.rep_dim, "enc1")
        self.W_e2 = self.create_weight(self.input_dim, self.rep_dim, "enc2")
        self.B_e = self.create_bias(self.rep_dim, "enc")

        self.W_d1 = self.create_weight(self.rep_dim, self.rep_dim, "dec1")
        self.W_d2 = self.create_weight(self.rep_dim, self.input_dim, "dec2")
        self.B_d1 = self.create_bias(self.rep_dim, "dec1")
        self.B_d2 = self.create_bias(self.input_dim, "dec2")

        self.h0 = None
        if self.additional_h:
            self.h0 = self.create_vector(self.output_dim, "h0")

        self.register_parameters(self.W_e1, self.W_e2, self.W_d1, self.W_d2,
                                 self.B_e, self.B_d1, self.B_d2)

