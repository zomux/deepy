#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy.utils import AutoEncoderCost
import theano.tensor as T

from network import NeuralNetwork

class AutoEncoder(NeuralNetwork):
    """
    Auto encoder.
    Must call stack_encoding before stack_decoding.
    Parameters:
        rep_dim - dimension of representation
    """
    def __init__(self, input_dim, rep_dim=None, config=None, input_tensor=None):
        super(AutoEncoder, self).__init__(input_dim, config=config, input_tensor=input_tensor)

        self.rep_dim = rep_dim
        self.encoding_layes = []
        self.decoding_layers = []
        self.encoding_network = None
        self.decoding_network = None

    def _cost_func(self, y):
        return AutoEncoderCost(self.input_variables[0], y).get()

    @property
    def cost(self):
        return self._cost_func(self.output)

    @property
    def test_cost(self):
        return self._cost_func(self.test_output)

    def stack_encoders(self, *layers):
        """
        Stack encoding layers, this must be done before stacking decoding layers.
        """
        self.stack(*layers)
        self.encoding_layes.extend(layers)

    def stack_decoders(self, *layers):
        """
        Stack decoding layers.
        """
        self.stack(*layers)
        self.decoding_layers.extend(layers)

    def encode(self, x):
        """
        Encode given input.
        """
        if not self.encoding_network:
            self.encoding_network = NeuralNetwork(self.input_dim, self.network_config, self.input_tensor)
            self.encoding_network.stack(*self.encoding_layes)
        self.encoding_network.compute(x)

    def decode(self, x):
        """
        Decode given representation.
        """
        if not self.rep_dim:
            raise Exception("rep_dim must be set to decode.")
        if not self.decoding_network:
            self.decoding_network = NeuralNetwork(self.rep_dim, self.network_config)
            self.decoding_network.stack(*self.decoding_layers)
        self.decoding_network.compute(x)
