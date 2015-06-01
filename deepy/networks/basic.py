#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import NeuralNetwork


class BasicNetwork(NeuralNetwork):
    """
    A simple neural network that the last layer outputs cost.
    """

    def __init__(self, input_dim, model=None, config=None, input_tensor=None):
        """
        Create a basic network.

        Parameters:
            input_dim - dimension of input variable
            model - a short hand to specify the model
            config - network configuration
            input_tensor - specify the tensor of input if it's special
        """
        super(BasicNetwork, self).__init__(input_dim, config=config, input_tensor=input_tensor)
        if model:
            self.stack(model)


    @property
    def cost(self):
        return self.output

    @property
    def test_cost(self):
        return self.test_output