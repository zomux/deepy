#!/usr/bin/env python
# -*- coding: utf-8 -*-

from layer import NeuralLayer


class Chain(NeuralLayer):
    """
    Stack many layers to form a chain.
    This is useful to reuse layers in a customized layer.
    Usage:
        As part of the main pipe line:
            chain = Chain(layer1, layer2)
            model.stack(chain)
        As part of the computational graph:
            chain = Chain(layer1, layer2)
            y = chain.compute(x)
    """

    def __init__(self, *layers):
        super(Chain, self).__init__("chain")
        self.layers = []
        self._layers_to_stack = []
        if len(layers) == 1 and type(layers[0]) == int:
            # This is a deprecated using of Chain
            self.input_dim = layers[0]
        else:
            self.stack(*layers)

    def stack(self, *layers):
        if self.input_dim is None or self.input_dim == 0:
            # Don't know the input dimension until connect
            self._layers_to_stack.extend(layers)
        else:
            self._register_layers(*layers)
        return self

    def _register_layers(self, *layers):
        for layer in layers:
            if not self.layers:
                layer.initialize(self.input_dim)
            else:
                layer.initialize(self.layers[-1].output_dim)
            self.layers.append(layer)
            self.output_dim = layer.output_dim
        self.register_inner_layers(*self.layers)

    def prepare(self, *layers):
        if self._layers_to_stack:
            self._register_layers(*self._layers_to_stack)
            self._layers_to_stack = []

    def compute_tensor(self, x):
        return self._output(x, False)

    def compute_test_tesnor(self, x):
        return self._output(x, True)

    def _output(self, x, test):
        y = x
        for layer in self.layers:
            y = layer.compute_flexible_tensor(y, test=test)
        return y
