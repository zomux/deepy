#!/usr/bin/env python
# -*- coding: utf-8 -*-

from layer import NeuralLayer


class Block(NeuralLayer):
    """
    Create a block, which contains the parameters of many connected layers.
    """

    def __init__(self):
        super(Block, self).__init__("block")
        self.layers = []

    def register(self, *layers):
        """
        Register many connected layers.
        :type layers: list of NeuralLayer
        """
        for layer in layers:
            self.register_layer(layer)

    def register_layer(self, layer):
        """
        Register one connected layer.
        :type layer: NeuralLayer
        """
        if not layer.connected:
            raise SystemError("%s is not connected, call `compute` before register it" % str(layer))
        self.layers.append(layer)
        self.register_inner_layers(layer)

    def output(self, x):
        return x

    def test_output(self, x):
        return x