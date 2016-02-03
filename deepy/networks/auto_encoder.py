#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy.utils import AutoEncoderCost
import theano.tensor as T

from network import NeuralNetwork

class AutoEncoder(NeuralNetwork):
    """
    A class for defining simple auto encoders.
    Must call stack_encoding before stack_decoding.
    """
    def __init__(self, input_dim, rep_dim, input_tensor=None):
        """
        Create an auto-encoder.
        :param input_dim: dimension of the input layer
        :param rep_dim: dimension of the representation layer
        :param input_tensor:
        :return:
        """
        super(AutoEncoder, self).__init__(input_dim, input_tensor=input_tensor)

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
            self.encoding_network = NeuralNetwork(self.input_dim, self.input_tensor)
            for layer in self.encoding_layes:
                self.encoding_network.stack_layer(layer, no_setup=True)
        return self.encoding_network.compute(x)

    def decode(self, x):
        """
        Decode given representation.
        """
        if not self.rep_dim:
            raise Exception("rep_dim must be set to decode.")
        if not self.decoding_network:
            self.decoding_network = NeuralNetwork(self.rep_dim)
            for layer in self.decoding_layers:
                self.decoding_network.stack_layer(layer, no_setup=True)
        return self.decoding_network.compute(x)
