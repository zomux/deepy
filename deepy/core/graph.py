#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
import gzip
from inspect import getargspec
from env import env
import theano
import logging as loggers
from deepy.utils.activations import get_activation
from decorations import neural_computation
from costs import RegressionCost, CrossEntropyCost, AutoEncoderCost
from disconnected_grad import disconnected_grad
logging = loggers.getLogger(__name__)


class GraphBuilder(object):
    """
    Tool for creating computational graph in deepy.
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
        if isinstance(theano_tensor, NeuralVariable):
            var = theano_tensor
            if dim != 0:
                var.output_dim = dim
        else:
            var = NeuralVariable(theano_tensor, dim=dim)
        if test_shape:
            if type(test_shape) != list and type(test_shape) != tuple:
                var.set_test_value(test_shape)
            else:
                var.set_test_value(env.numpy_rand.rand(*test_shape).astype(dtype))
        return var

    def create_vars_from_data(self, dataset):
        """
        Create vars given a dataset and set test values.
        Useful when dataset is already defined.
        """

    @neural_computation
    def activation(self, x, name='tanh'):
        """
        Compute an activation value.
        """
        return get_activation(name)(x)

    @neural_computation
    def cross_entropy_cost(self, y, target_index):
        return CrossEntropyCost(y, target_index).get()

    @neural_computation
    def least_squares_cost(self, y, target):
        return RegressionCost(y, target).get()

    @neural_computation
    def auto_encoder_cost(self, y, target):
        return AutoEncoderCost(y, target).get()

    @neural_computation
    def shared(self, value, name=None):
        """
        Create a shared theano scalar value.
        """
        if type(value) == int:
            final_value = np.array(value, dtype="int32")
        elif type(value) == float:
            final_value = np.array(value, dtype=env.FLOATX)
        else:
            final_value = value

        return theano.shared(final_value, name=name)

    @neural_computation
    def disconnect(self, x):
        """
        Disconnect a variable from backpropagation.
        """
        return disconnected_grad(x)

    def compile(self, input_dim=0, model=None, input_tensor=None, monitors=None,
                 cost=None, output=None, outputs=None, blocks=None, input_vars=None, target_vars=None, output_map=None):
        from comp_graph import ComputationalGraph
        # Pass the arguments to `ComputationalGraph`
        args = [arg for arg in getargspec(GraphBuilder.compile).args if arg != "self"]
        arg_vals = [locals()[k] for k in args]
        kwargs = dict(zip(args, arg_vals))
        return ComputationalGraph(**kwargs)

    def fill_parameters(self, path, blocks, exclude_free_params=False, check_parameters=False):
        """
        Load parameters from file to fill all blocks sequentially.
        :type blocks: list of deepy.layers.Block
        """
        if not os.path.exists(path):
            raise Exception("model {} does not exist".format(path))
        # Decide which parameters to load
        normal_params = sum([nn.parameters for nn in blocks], [])
        all_params = sum([nn.all_parameters for nn in blocks], [])
        # Load parameters
        if path.endswith(".gz"):
            opener = gzip.open if path.lower().endswith('.gz') else open
            handle = opener(path, 'rb')
            saved_params = pickle.load(handle)
            handle.close()
            # Write parameters
            if len(all_params) != len(saved_params):
                logging.warning(
                    "parameters in the network: {}, parameters in the dumped model: {}".format(len(all_params),
                                                                                               len(saved_params)))
            for target, source in zip(all_params, saved_params):
                if not exclude_free_params or target not in normal_params:
                    target.set_value(source)
        elif path.endswith(".npz"):
            arrs = np.load(path)
            # Write parameters
            if len(all_params) != len(arrs.keys()):
                logging.warning(
                    "parameters in the network: {}, parameters in the dumped model: {}".format(len(all_params),
                                                                                               len(arrs.keys())))
            for target, idx in zip(all_params, range(len(arrs.keys()))):
                if not exclude_free_params or target not in normal_params:
                    source = arrs['arr_%d' % idx]
                    target.set_value(source)
        else:
            raise Exception("File format of %s is not supported, use '.gz' or '.npz' or '.uncompressed.gz'" % path)


if "graph" not in globals():
    graph = GraphBuilder()
