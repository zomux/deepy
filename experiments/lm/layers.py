#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from deepy.layers import NeuralLayer, Softmax3D, Softmax, Dense, Chain
from deepy.utils import CrossEntropyCost

from cost import LMCost

class FullOutputLayer(NeuralLayer):

    def __init__(self, vocab_size):
        super(FullOutputLayer, self).__init__("full_output")
        self.vocab_size = vocab_size


    def setup(self):
        self.core = Chain(self.input_dim).stack(Dense(self.vocab_size),
                                                Softmax3D())
        self.register_inner_layers(self.core)

    def output(self, x):
        return self.core.output(x)

class ClassOutputLayer(NeuralLayer):

    def __init__(self, output_size, class_size):
        super(ClassOutputLayer, self).__init__("class_output")
        self.output_size = output_size
        self.class_size = class_size

    def setup(self):
        # Output layers
        self.output_layer = Chain(self.input_dim).stack(Dense(self.output_size * self.class_size))
        self.softmax_layer = Softmax().connect(input_dim=self.output_size)

        self.class_layer = Chain(self.input_dim).stack(Dense(self.class_size),
                                                        Softmax3D())
        self.register_inner_layers(self.class_layer, self.output_layer)
        # Target tensor
        self.target_tensor = T.imatrix('target')
        self.register_external_targets(self.target_tensor)
        # arange cache
        self.arange_cache = theano.shared(np.arange(10*64), name="arange_cache")


    def output(self, x):
        """
        :param x: (batch, time, vec)
        """
        # Target class
        class_matrix = self.target_tensor // self.output_size
        class_vector = class_matrix.reshape((-1,))
        # Target index
        target_matrix = self.target_tensor % self.output_size
        target_vector = target_matrix.reshape((-1,))
        # Input matrix
        input_matrix = x.reshape((-1, self.input_dim))
        # Output matrix
        output_tensor3d = self.output_layer.output(x)
        output_matrix = output_tensor3d.reshape((-1, self.class_size, self.output_size))
        arange_vec = self.arange_cache[:output_matrix.shape[0]]
        sub_output_matrix = output_matrix[arange_vec, class_vector]
        # Softmax
        softmax_output_matrix = self.softmax_layer.output(sub_output_matrix)
        # Class prediction
        class_output_matrix = self.class_layer.output(x)
        # Costs
        output_cost = LMCost(softmax_output_matrix, target_vector).get()
        class_cost = LMCost(class_output_matrix, class_matrix).get()
        final_cost = output_cost + class_cost

        return final_cost

