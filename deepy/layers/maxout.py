#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano.tensor as T
from . import NeuralLayer
from dense import Dense

class Maxout(NeuralLayer):
    """
    Maxout activation unit.
     - http://arxiv.org/pdf/1302.4389.pdf
    """
    def __init__(self, output_dim=None, num_pieces=4, init=None, linear_transform=True):
        """
        :param num_pieces: pieces of sub maps
        """
        super(Maxout, self).__init__("maxout")
        self.num_pieces = num_pieces
        self.output_dim = output_dim
        self.linear_transform = linear_transform
        self.init = init


    def prepare(self):
        if self.output_dim is None:
            self.output_dim = self.input_dim // self.num_pieces
        if self.linear_transform:
            self.transformer = Dense(self.output_dim * self.num_pieces).connect(self.input_dim)
            self.register(self.transformer)


    def output(self, x):
        if self.linear_transform:
            x = self.transformer.output(x)
        # x ~ batch, time, size / batch, size
        new_shape = [x.shape[i] for i in range(x.ndim - 1)] + [self.output_dim, self.num_pieces]
        # new_shape ~ batch, time, out_dim, pieces / batch, out_dim, pieces
        output = T.max(x.reshape(new_shape, ndim=x.ndim + 1), axis=x.ndim)
        return output