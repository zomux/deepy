#!/usr/bin/env python
# -*- coding: utf-8 -*-

from layer import NeuralLayer


class Chain(NeuralLayer):
    """
    Stack many layers to form a chain.
    This is useful to reuse layers in a customized layer.
    Usage:
        As part of the main pipe line:
            chain = Chain().stack(layer1, layer2)
            model.stack(chain)
        As part of the computational graph:
            chain = Chain(input_dim).stack(layer1, layer2)
            y = chain.output(x)
    """

    def __init__(self, input_dim=None):
        super(Chain, self).__init__("chain")
        self.layers = []
        self.input_dim = input_dim
        self._layers_to_stack = []

    def stack(self, *layers):
        if self.input_dim == None:
            # Don't know the input dimension until connect
            self._layers_to_stack.extend(layers)
        else:
            self._register_layers(*layers)
        return self

    def _register_layers(self, *layers):
        for layer in layers:
            if not self.layers:
                layer.connect(self.input_dim)
            else:
                layer.connect(self.layers[-1].output_dim)
            self.layers.append(layer)
            self.output_dim = layer.output_dim
        self.register_inner_layers(*self.layers)

    def setup(self, *layers):
        if self._layers_to_stack:
            self._register_layers(*self._layers_to_stack)
            self._layers_to_stack = []

    def output(self, x):
        return self._output(x, False)

    def test_output(self, x):
        return self._output(x, True)

    def _output(self, x, test):
        y = x
        for layer in self.layers:
            y = layer.call(y, test=test)
        return y
