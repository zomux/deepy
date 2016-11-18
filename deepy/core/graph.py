#!/usr/bin/env python
# -*- coding: utf-8 -*-

from inspect import getargspec

from env import env


class GraphBuilder(object):
    """
    Tool for creating computational graph in deepy.
    """

    def vars_in_data(self, dataset):
        """
        Create vars given a dataset and set test values.
        Useful when dataset is already defined.
        """

    def block(self, name=None):
        """
        Create a block.
        """
        from deepy.layers.block import Block
        return Block(name)

    def var(self, theano_tensor, dim=0, dtype=env.FLOATX, test_shape=None):
        """
        Wrap a Theano tensor into the variable for defining neural network.
        :param dim: last dimension of tensor, 0 indicates that the last dimension is flexible
        :rtype: TensorVar
        """
        from deepy.core.neural_var import NeuralVariable
        var = NeuralVariable(theano_tensor, dim=dim)
        if test_shape:
            if type(test_shape) != list and type(test_shape) != tuple:
                var.set_test_value(test_shape)
            else:
                var.set_test_value(env.numpy_rand.rand(*test_shape).astype(dtype))
        return var


    def compile(self, input_dim=0, model=None, input_tensor=None, monitors=None,
                 cost=None, output=None, outputs=None, blocks=None, input_vars=None, target_vars=None, output_map=None):
        from comp_graph import ComputationalGraph
        # Pass the arguments to `ComputationalGraph`
        args = [arg for arg in getargspec(GraphBuilder.compile).args if arg != "self"]
        arg_vals = [locals()[k] for k in args]
        kwargs = dict(zip(args, arg_vals))
        return ComputationalGraph(**kwargs)


if "graph" not in globals():
    graph = GraphBuilder()
