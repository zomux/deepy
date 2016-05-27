#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging as loggers
import gzip
import cPickle as pickle
import os
from threading import Thread

import numpy as np
import theano.tensor as T
import theano
import filelock

import deepy
from deepy.layers.layer import NeuralLayer
from deepy.layers.block import Block
from deepy.utils import dim_to_var, TrainLogger

logging = loggers.getLogger(__name__)

DEEPY_MESSAGE = "deepy version = %s" % deepy.__version__


def save_network_params(params, path):
    lock = filelock.FileLock(path)
    with lock:
        if path.endswith('gz'):
            opener = gzip.open if path.lower().endswith('.gz') else open
            handle = opener(path, 'wb')
            pickle.dump(params, handle)
            handle.close()
        elif path.endswith('uncompressed.npz'):
            np.savez(path, *params)
        elif path.endswith('.npz'):
            np.savez_compressed(path, *params)
        else:
            raise Exception("File format of %s is not supported, use '.gz' or '.npz' or '.uncompressed.gz'" % path)

class NeuralNetwork(object):
    """
    The base class of neural networks.
    """

    def __init__(self, input_dim, input_tensor=None):
        logging.info(DEEPY_MESSAGE)
        self.input_dim = input_dim
        self.input_tensor = input_tensor
        self.parameter_count = 0

        self.parameters = []
        self.free_parameters = []

        self.training_updates = []
        self.updates = []

        self.input_variables = []
        self.target_variables = []

        self.training_callbacks = []
        self.testing_callbacks = []
        self.epoch_callbacks = []

        self.layers = []

        self._hidden_outputs = []
        self.training_monitors = []
        self.testing_monitors = []

        self.setup_variables()
        self.train_logger = TrainLogger()

    def stack_layer(self, layer, no_setup=False):
        """
        Stack a neural layer.
        :type layer: NeuralLayer
        :param no_setup: whether the layer is already initialized
        """
        if layer.name:
            layer.name += "%d" % (len(self.layers) + 1)
        if not self.layers:
            layer.initialize(self.input_dim, no_prepare=no_setup)
        else:
            layer.initialize(self.layers[-1].output_dim, no_prepare=no_setup)
        self._output = layer.compute_tensor(self._output)
        self._test_output = layer.compute_test_tesnor(self._test_output)
        self._hidden_outputs.append(self._output)
        self.register_layer(layer)
        self.layers.append(layer)

    def register(self, *layers):
        """
        Register multiple layers as the components of the network.
        The parameter of those layers will be trained.
        But the output of the layer will not be stacked.
        """
        for layer in layers:
            self.register_layer(layer)

    def register_layer(self, layer):
        """
        Register the layer so that it's param will be trained.
        But the output of the layer will not be stacked.
        """
        if type(layer) == Block:
            layer.fix()
        self.parameter_count += layer.parameter_count
        self.parameters.extend(layer.parameters)
        self.free_parameters.extend(layer.free_parameters)
        self.training_monitors.extend(layer.training_monitors)
        self.testing_monitors.extend(layer.testing_monitors)
        self.updates.extend(layer.updates)
        self.training_updates.extend(layer.training_updates)
        self.input_variables.extend(layer.external_inputs)
        self.target_variables.extend(layer.external_targets)

        self.training_callbacks.extend(layer.training_callbacks)
        self.testing_callbacks.extend(layer.testing_callbacks)
        self.epoch_callbacks.extend(layer.epoch_callbacks)

    def first_layer(self):
        """
        Return first layer.
        """
        return self.layers[0] if self.layers else None

    def stack(self, *layers):
        """
        Stack layers.
        """
        for layer in layers:
            self.stack_layer(layer)
        return self

    def prepare_training(self):
        """
        This function will be called before training.
        """
        self.report()

    def monitor_layer_outputs(self):
        """
        Monitoring the outputs of each layer.
        Useful for troubleshooting convergence problems.
        """
        for layer, hidden in zip(self.layers, self._hidden_outputs):
            self.training_monitors.append(('mean(%s)' % (layer.name), abs(hidden).mean()))

    @property
    def all_parameters(self):
        """
        Return all parameters.
        """
        params = []
        params.extend(self.parameters)
        params.extend(self.free_parameters)

        return params

    def setup_variables(self):
        """
        Set up variables.
        """
        if self.input_tensor:
            if type(self.input_tensor) == int:
                x = dim_to_var(self.input_tensor, name="x")
            else:
                x = self.input_tensor
        else:
            x = T.matrix('x')
        self.input_variables.append(x)
        self._output = x
        self._test_output = x

    def _compile(self):
        if not hasattr(self, '_compute'):
            self._compute = theano.function(
                filter(lambda x: x not in self.target_variables, self.input_variables),
                self.test_output, updates=self.updates, allow_input_downcast=True)

    def compute(self, *x):
        """
        Return network output.
        """
        self._compile()
        return self._compute(*x)

    @property
    def output(self):
        """
        Return output variable.
        """
        return self._output

    @property
    def test_output(self):
        """
        Return output variable in test time.
        """
        return self._test_output

    @property
    def cost(self):
        """
        Return cost variable.
        """
        return T.constant(0)

    @property
    def test_cost(self):
        """
        Return cost variable in test time.
        """
        return self.cost

    def save_params(self, path, new_thread=False):
        """
        Save parameters to file.
        """
        logging.info("saving parameters to %s" % path)
        param_variables = self.all_parameters
        params = [p.get_value().copy() for p in param_variables]
        if new_thread:
            thread = Thread(target=save_network_params, args=(params, path))
            thread.start()
        else:
            save_network_params(params, path)
        self.train_logger.save(path)

    def load_params(self, path, exclude_free_params=False):
        """
        Load parameters from file.
        """
        if not os.path.exists(path): return;
        logging.info("loading parameters from %s" % path)
        # Decide which parameters to load
        if exclude_free_params:
            params_to_load = self.parameters
        else:
            params_to_load = self.all_parameters
        # Load parameters
        if path.endswith(".gz"):
            opener = gzip.open if path.lower().endswith('.gz') else open
            handle = opener(path, 'rb')
            saved_params = pickle.load(handle)
            handle.close()
            # Write parameters
            for target, source in zip(params_to_load, saved_params):
                logging.info('%s: setting value %s', target.name, source.shape)
                target.set_value(source)
        elif path.endswith(".npz"):
            arrs = np.load(path)
            # Write parameters
            for target, idx in zip(params_to_load, range(len(arrs.keys()))):
                source = arrs['arr_%d' % idx]
                logging.info('%s: setting value %s', target.name, source.shape)
                target.set_value(source)
        else:
            raise Exception("File format of %s is not supported, use '.gz' or '.npz' or '.uncompressed.gz'" % path)

        self.train_logger.load(path)

    def report(self):
        """
        Print network statistics.
        """
        logging.info("network inputs: %s", " ".join(map(str, self.input_variables)))
        logging.info("network targets: %s", " ".join(map(str, self.target_variables)))
        logging.info("network parameters: %s", " ".join(map(str, self.all_parameters)))
        logging.info("parameter count: %d", self.parameter_count)

    def epoch_callback(self):
        """
        Callback for each epoch.
        """
        for cb in self.epoch_callbacks:
            cb()

    def training_callback(self):
        """
        Callback for each training iteration.
        """
        for cb in self.training_callbacks:
            cb()

    def testing_callback(self):
        """
        Callback for each testing iteration.
        """
        for cb in self.training_callbacks:
            cb()
