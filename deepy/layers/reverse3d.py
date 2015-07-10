#!/usr/bin/env python
# -*- coding: utf-8 -*-

from layer import NeuralLayer
import theano
import theano.tensor as T

class Reverse3D(NeuralLayer):

    def __init__(self):
        super(Reverse3D, self).__init__("reverse3d")

    def output(self, x):
        return x[:, ::-1, :]
