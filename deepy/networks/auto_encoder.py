#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import theano.tensor as T
from deepy.networks.basic_nn import NeuralNetwork


class AutoEncoder(NeuralNetwork):

    @property
    def cost(self):
        err = self.vars.y - self.vars.x
        return T.mean((err * err).sum(axis=1))

    # def encode(self, x, layer=None):
    #     enc = self.feed_forward(x)[(layer or len(self.layers) // 2) - 1]
    #     return enc
    #
    # def decode(self, z, layer=None):
    #     if not hasattr(self, '_decoders'):
    #         self._decoders = {}
    #     layer = layer or len(self.layers) // 2
    #     if layer not in self._decoders:
    #         self._decoders[layer] = theano.function(
    #             [self.hiddens[layer - 1]], [self.y], updates=self.updates)
    #     return self._decoders[layer](z)[0]