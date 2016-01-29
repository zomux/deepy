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
        self.fixed = False

    def fix(self):
        """
        Fix the block, register all the parameters of sub layers.
        :return:
        """
        if not self.fixed:
            for layer in self.layers:
                if not layer.initialized:
                    raise Exception("All sub layers in a block must be initialized when fixing it.")
                self.register_inner_layers(layer)
            self.fixed = True


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
        if self.fixed:
            raise Exception("After a block is fixed, no more layers can be registered.")
        self.layers.append(layer)


    def compute_tensor(self, x):
        return x

    def compute_test_tesnor(self, x):
        return x