#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano.tensor as T
from . import NeuralLayer

class Maxout(NeuralLayer):
    """
    Maxout activation unit.
     - http://arxiv.org/pdf/1302.4389.pdf
    """
    def __init__(self, output_dim, num_pieces=4, init=None):
        """
        :param num_pieces: pieces of sub maps
        """
        super(Maxout, self).__init__("maxout")
        self.num_pieces = num_pieces
        self.output_dim = output_dim
        self.init = init


    def setup(self):
        self.W = self.create_weight(shape=(self.num_pieces, self.input_dim, self.output_dim),
                                    initializer=self.init, suffix="maxout")
        self.B = self.create_bias(shape=(self.num_pieces, self.output_dim), suffix="maxout")
        self.register_parameters(self.W, self.B)


    def output(self, x):
        return T.max(T.dot(x, self.W) + self.B, axis=1)