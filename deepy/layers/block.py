#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy.layers.layer import NeuralLayer

class Block(NeuralLayer):
    """
    Create a block, which contains the parameters of many connected layers.
    """

    _BLOCK_COUNT = 0

    def __init__(self, *layers, **kwargs):
        """
        Create a new parameter block with some layers.
        You can also specify a name through kwargs.
        """
        name = kwargs['name'] if 'name' in kwargs else "block_{}".format(self._BLOCK_COUNT + 1)
        super(Block, self).__init__(name)
        self._BLOCK_COUNT += 1
        self.layers = list(layers)
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

    def add_parameters(self, *parameters):
        from deepy.core.neural_var import NeuralVariable
        for param in parameters:
            if isinstance(param, NeuralVariable):
                param = param.tensor
            self.parameters.append(param)

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

    def load_params(self, path, exclude_free_params=False):
        from deepy.core import graph
        """
        Load parameters to the block.
        """
        from deepy.core.comp_graph import ComputationalGraph
        model = graph.compile(blocks=[self])
        model.load_params(path, exclude_free_params=exclude_free_params)

    @property
    def all_parameters(self):
        return self.parameters