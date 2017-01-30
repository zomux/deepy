#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy.networks import NeuralNetwork


class ComputationalGraph(NeuralNetwork):
    """
    A class for defining computational graphs.
    This class can be used to design very complex models, such as Neural Turing Machine.
    """

    def __init__(self, input_dim=0, model=None, input_tensor=None, monitors=None,
                 cost=None, output=None, outputs=None, blocks=None, input_vars=None, target_vars=None, updates=None):
        """
        Create a basic network.

        Parameters:
            input_dim - dimension of input variable
            model - a short hand to specify the model
            config - network configuration
            input_tensor - specify the tensor of input if it's special
        """
        from deepy.core.neural_var import NeuralVariable
        from deepy.core.tensor_conversion import convert_to_theano_var
        from theano.sandbox.cuda import CudaNdarraySharedVariable
        super(ComputationalGraph, self).__init__(input_dim, input_tensor=input_tensor)
        self.input_variables = []
        self.target_variables = []
        if model:
            self.stack(model)
        if cost:
            self.stack(cost)
        if output:
            if cost:
                self._test_output = output.tensor
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
            self._test_outputs, _, _ = convert_to_theano_var(outputs)
        if updates:
            if isinstance(updates, dict):
                updates = updates.items()
            update_tensors, _, _ = convert_to_theano_var(updates)
            self.updates.extend(update_tensors)

        if monitors:
            if type(monitors) == dict:
                monitors = monitors.items()
            for monitor in monitors:
                if type(monitor) != tuple:
                    raise Exception("monitors shall be tuples of (name, var).")
                name, var = monitor
                if isinstance(var, NeuralVariable):
                    var = var.tensor
                if isinstance(var, CudaNdarraySharedVariable):
                    var *= 1.0  # Avoid CudaNdarray
                self.training_monitors.append((name, var))
                self.testing_monitors.append((name, var))


    @property
    def cost(self):
        return self.output

    @property
    def test_cost(self):
        return self.test_output