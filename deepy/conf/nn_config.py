#!/usr/bin/env python
# -*- coding: utf-8 -*-



class NetworkConfig(object):

    def __init__(self, input_size):
        """
        Create a config for neural network
        :param input_size: size of input vector
        :return:
        """
        self.input_size = input_size
        # :type: list of deepy.NeuralLayer
        self.layers = []
        self.no_learn_biases = False

        # Noise
        self.input_noise = 0.
        self.input_dropouts = 0.