#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy.layers import NeuralLayer
from deepy.utils import onehot_tensor

class OneHotEmbedding(NeuralLayer):
    """
    One-hot embedding layer.
    Computation: [0,1,2]  ---> [[1,0,0],[0,1,0],[0,0,1]]
    """
    def __init__(self, vocab_size):
        super(OneHotEmbedding, self).__init__("onehot")
        self.vocab_size = vocab_size
        self.output_dim = vocab_size

    def output(self, x):
        return onehot_tensor(x, self.vocab_size)