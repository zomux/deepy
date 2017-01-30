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
import logging as loggers
from tensor_conversion import neural_computation
from disconnected_grad import disconnected_grad
from deepy.utils import Scanner
logging = loggers.getLogger(__name__)


class GraphBuilder(object):
    """
    Tool for creating computational graph in deepy.
    """

    def __init__(self):
        self._default_block = self.new_block("default_block")

    def default_block(self):
        """
        Return the default block.
        """
        return self._default_block

    def collect_parameters(self):
        """
        Return the default block, as all parameters will be registered to the default one.
        """
        return self._default_block

    def new_block(self, *layers, **kwargs):
        """
        Create a parameters block.
        :param layers: register some layers in the block
        :param name: specify the name of this block
        """
        from deepy.layers.block import Block
        block = Block(*layers, **kwargs)
        return block

    def var(self, tensor_type, last_dim=0, test_shape=None):
        """
        An alias of deepy.tensor.var.
        """
        from deepy.tensor import var
        return var(tensor_type, last_dim=last_dim, test_shape=test_shape)

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
            if numpy_tensor.dtype == "int64":
                numpy_tensor = numpy_tensor.astype("int32")
            if numpy_tensor.dtype == "float64":
                numpy_tensor = numpy_tensor.astype(env.FLOATX)
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
    def scan(self, func, sequences=None, outputs=None, non_sequences=None, block=None, **kwargs):
        """
        A loop function, the usage is identical with the theano one.
        :type block: deepy.layers.Block
        """
        results, updates = Scanner(func, sequences, outputs, non_sequences, neural_computation=True, **kwargs).compute()
        if block and updates:
            if type(updates) == dict:
                updates = updates.items()
            block.register_updates(*updates)
        return results

    def loop(self, sequences=None, outputs=None, non_sequences=None, block=None, **kwargs):
        """
        Start a loop.
        Usage:
        ```
        with deepy.graph.loop(sequences={"x": x}, outputs={"o": None}) as vars:
            vars.o = vars.x + 1
        loop_outputs = deepy.graph.loop_outputs()
        result = loop_outputs.o
        ```
        """
        from loop import Loop
        return Loop(sequences, outputs, non_sequences, block, **kwargs)

    def get_trainer(self, model,  method='sgd', config=None, annealer=None, validator=None):
        """
        Get a trainer to optimize given model.
        :rtype: deepy.trainers.GeneralNeuralTrainer
        """
        from deepy.trainers import GeneralNeuralTrainer
        return GeneralNeuralTrainer(model, method=method, config=config, annealer=annealer, validator=validator)

    def shared(self, value, name=None):
        """
        Create a shared theano scalar value.
        """
        from neural_var import NeuralVariable
        if type(value) == int:
            final_value = np.array(value, dtype="int32")
        elif type(value) == float:
            final_value = np.array(value, dtype=env.FLOATX)
        else:
            final_value = value

        last_dim = final_value.shape[-1] if hasattr(final_value, 'shape') and final_value.shape else 0
        return NeuralVariable(theano.shared(final_value, name=name), last_dim)

    @neural_computation
    def disconnect(self, x):
        """
        Disconnect a variable from backpropagation.
        """
        return disconnected_grad(x)

    def compile(self, input_dim=0, model=None, input_tensor=None, monitors=None,
                 cost=None, output=None, outputs=None, blocks=None, input_vars=None, target_vars=None, updates=None):
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
