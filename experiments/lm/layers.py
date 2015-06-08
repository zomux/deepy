#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy.layers import NeuralLayer, Softmax3D, Softmax, Dense, Chain
from deepy.utils import CrossEntropyCost

import theano
import theano.tensor as T

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
        self.output_layer = Chain(self.input_dim).stack(Dense(self.output_size * self.class_size))
        self.softmax_layer = Softmax().connect(input_dim=self.output_size)

        self.class_layer = Chain(self.input_dim).stack(Dense(self.class_size),
                                                        Softmax3D())
        self.register_inner_layers(self.class_layer, self.output_layer)
        self.target_tensor = T.imatrix('target')
        self.register_external_targets(self.target_tensor)

    def _output_step(self, output_vec, class_scalar):
        start_index = class_scalar * self.output_size
        return output_vec[start_index: start_index + self.output_size]

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
        output_matrix = self.output_layer.output(input_matrix)
        sub_output_matrix, _ = theano.map(self._output_step,
                                      sequences=[output_matrix, class_vector])
        # Softmax
        softmax_output_matrix = self.softmax_layer.output(sub_output_matrix)
        # Class prediction
        class_output_matrix = self.class_layer.output(x)
        # Costs
        output_cost = LMCost(softmax_output_matrix, target_vector).get()
        class_cost = LMCost(class_output_matrix, class_matrix).get()
        final_cost = output_cost + class_cost

        return final_cost

