#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import NeuralNetwork


class ComputationalGraph(NeuralNetwork):
    """
    A class for defining computational graphs.
    This class can be used to design very complex models, such as Neural Turing Machine.
    """

    def __init__(self, input_dim=0, model=None, input_tensor=None, monitors=None,
                 cost=None, output=None, outputs=None, blocks=None, input_vars=None, target_vars=None, output_map=None):
        """
        Create a basic network.

        Parameters:
            input_dim - dimension of input variable
            model - a short hand to specify the model
            config - network configuration
            input_tensor - specify the tensor of input if it's special
        """
        super(ComputationalGraph, self).__init__(input_dim, input_tensor=input_tensor)
        if model:
            self.stack(model)
        if cost:
            self.stack(cost)
        if output:
            if cost:
                self._test_output = output.test_tensor
            else:
                self.stack(output)
        if blocks:
            self.register(*blocks)
        if input_vars:
            self.input_variables = [t.tensor for t in input_vars]
        if target_vars:
            self.target_variables = [t.tensor for t in target_vars]
        if outputs:
            if not output and not cost:
                self._test_output = None
            self._test_outputs = [o.test_tensor for o in outputs]

        self.output_map = output_map if output_map else {}

        if monitors:
            if type(monitors) == dict:
                monitors = monitors.items()
            for monitor in monitors:
                if type(monitor) != tuple:
                    raise Exception("monitors shall be tuples of (name, var).")
                name, var = monitor
                self.training_monitors.append((name, var.tensor))
                self.testing_monitors.append((name, var.test_tensor))


    @property
    def cost(self):
        return self.output

    @property
    def test_cost(self):
        return self.test_output

graph = ComputationalGraph