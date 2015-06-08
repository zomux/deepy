#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging as loggers
import gzip
import cPickle as pickle
import os

import theano.tensor as T
import theano

import deepy
from deepy.layers.layer import NeuralLayer
from deepy.conf import NetworkConfig
from deepy.utils import dim_to_var, TrainLogger

logging = loggers.getLogger(__name__)

DEEPY_MESSAGE = "deepy version = %s" % deepy.__version__

class NeuralNetwork(object):
    """
    Basic neural network class.
    """

    def __init__(self, input_dim, config=None, input_tensor=None):
        logging.info(DEEPY_MESSAGE)
        self.network_config = config if config else NetworkConfig()
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

        if self.network_config.layers:
            self.stack(self.network_config.layers)

    def stack_layer(self, layer):
        """
        Stack a neural layer.
        :type layer: NeuralLayer
        """
        layer.name += "%d" % (len(self.layers) + 1)
        if not self.layers:
            layer.connect(self.input_dim, network_config=self.network_config)
        else:
            layer.connect(self.layers[-1].output_dim, previous_layer=self.layers[-1], network_config=self.network_config)
        self._output = layer.output(self._output)
        self._test_output = layer.test_output(self._test_output)
        self._hidden_outputs.append(self._output)
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
        self.layers.append(layer)

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

    def save_params(self, path):
        """
        Save parameters to file.
        """
        logging.info("saving parameters to %s" % path)
        opener = gzip.open if path.lower().endswith('.gz') else open
        handle = opener(path, 'wb')
        pickle.dump([p.get_value().copy() for p in self.all_parameters], handle)
        handle.close()
        self.train_logger.save(path)

    def load_params(self, path):
        """
        Load parameters from file.
        """
        if not os.path.exists(path): return;
        logging.info("loading parameters from %s" % path)
        opener = gzip.open if path.lower().endswith('.gz') else open
        handle = opener(path, 'rb')
        saved = pickle.load(handle)
        for target, source in zip(self.all_parameters, saved):
            logging.info('%s: setting value %s', target.name, source.shape)
            target.set_value(source)
        handle.close()
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