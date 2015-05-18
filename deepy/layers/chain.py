#!/usr/bin/env python
# -*- coding: utf-8 -*-

from layer import NeuralLayer


class Chain(NeuralLayer):
    """
    Stack many layers to form a chain.
    This is useful to reuse layers in a customized layer.
    """

    def __init__(self, input_dim):
        super(Chain, self).__init__("chain")
        self.layers = []
        self.input_dim = input_dim

    def stack(self, *layers):
        for layer in layers:
            if not self.layers:
                layer.connect(self.input_dim)
            else:
                layer.connect(self.layers[-1].output_dim)
            self.layers.append(layer)
            self.output_dim = layer.output_dim
        self.register_inner_layers(*self.layers)
        return self

    def output(self, x):
        return self._output(x, False)

    def test_output(self, x):
        return self._output(x, True)

    def _output(self, x, test):
        y = x
        for layer in self.layers:
            y = layer.call(y, test=test)
        return y