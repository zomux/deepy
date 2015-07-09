#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano.tensor as T
from deepy.layers import NeuralLayer
from deepy.utils import FLOATX

class WordEmbedding(NeuralLayer):
    """
    Word embedding layer.
    The word embeddings are randomly initialized, and are learned over the time.
    """
    def __init__(self, size, vocab_size, zero_index=None):
        super(OneHotEmbedding, self).__init__("onehot")
        self.size = size
        self.vocab_size = vocab_size
        self.output_dim = size
        self.zero_index = zero_index

    def setup(self):
        self.embed_matrix = self.create_weight(self.vocab_size, self.size, "embed")
        self.register_parameters(self.embed_matrix)

    def output(self, x):
        ret_tensor = self.embed_matrix[x.flatten()].reshape((x.shape[0], x.shape[1], self.size))
        if self.zero_index != None:
            mask = T.neq(x, self.zero_index)
            ret_tensor *= mask[:, :, None]
        return ret_tensor
