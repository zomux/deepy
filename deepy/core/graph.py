#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
import gzip
from inspect import getargspec
from env import env
import theano
import theano.tensor as TT
from theano.tensor.var import TensorVariable
import logging as loggers
from deepy.utils.activations import get_activation
from decorations import neural_computation
from costs import RegressionCost, CrossEntropyCost, AutoEncoderCost
from disconnected_grad import disconnected_grad
from deepy.utils import Scanner
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

    def var(self, tensor_type, last_dim=0, test_shape=None):
        """
        Wrap a Theano tensor into the variable for defining neural network.
        :param last_dim: last dimension of tensor, 0 indicates that the last dimension is flexible
        :rtype: TensorVar
        """
        # Create tensor
        from deepy.core.neural_var import NeuralVariable
        if isinstance(tensor_type, NeuralVariable):
            var = tensor_type
            if last_dim != 0:
                var.output_dim = last_dim
        elif isinstance(tensor_type, TensorVariable):
            var = NeuralVariable(tensor_type, dim=last_dim)
        elif isinstance(tensor_type, str):
            theano_tensor = getattr(TT, tensor_type)()
            var = NeuralVariable(theano_tensor, dim=last_dim)
        else:
            raise Exception("tensor_type shall be a string or a NeuralVariable")
        # Create test value
        if test_shape:
            if type(test_shape) != list and type(test_shape) != tuple:
                var.set_test_value(test_shape)
            else:
                var.set_test_value(env.numpy_rand.rand(*test_shape).astype(var.tensor.dtype))
        return var

    def create_vars_from_data(self, dataset, split="train"):
        """
        Create vars given a dataset and set test values.
        Useful when dataset is already defined.
        """
        from deepy.core.neural_var import NeuralVariable
        vars = []
        if split == "valid":
            data_split = dataset.valid_set()
        elif split == "test":
            data_split = dataset.test_set()
        else:
            data_split = dataset.train_set()
        first_data_piece = list(data_split)[0]
        for i, numpy_tensor in enumerate(first_data_piece):
            type_map = {
                0: "scalar",
                1: "vector",
                2: "matrix",
                3: "tensor3",
                4: "tensor4",
                5: "tensor5",
            }
            tensor_type = type_map[numpy_tensor.ndim] if numpy_tensor.ndim in type_map else type_map[0]
            if numpy_tensor.dtype.kind == "i":
                tensor_type = "i" + tensor_type
            theano_tensor = getattr(TT, tensor_type)("input_{}_{}".format(i + 1, tensor_type))
            last_dim = numpy_tensor.shape[-1]
            var = NeuralVariable(theano_tensor, dim=last_dim)
            var.set_test_value(numpy_tensor)
            vars.append(var)
        return vars



    @neural_computation
    def activation(self, x, name='tanh'):
        """
        Compute an activation value.
        """
        return get_activation(name)(x)

    @neural_computation
    def scan(self, func, sequences=None, outputs_info=None, non_sequences=None, **kwargs):
        """
        A loop function, the usage is identical with the theano one.
        """
        return Scanner(func, sequences, outputs_info, non_sequences, **kwargs).compute()

    def get_trainer(self, model, config=None, method='sgd'):
        """
        Get a trainer to optimize given model.
        :rtype: deepy.trainers.GeneralNeuralTrainer
        """
        from deepy.trainers import GeneralNeuralTrainer
        return GeneralNeuralTrainer(model, config=config, method=method)

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
