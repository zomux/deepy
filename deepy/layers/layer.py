#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging as loggers

import numpy as np
import theano

from deepy.util.functions import FLOATX, global_rand

logging = loggers.getLogger(__name__)

class NeuralLayer(object):

    def __init__(self, name="unknown"):
        """
        Create a neural layer.
        """
        self.name = name
        self.input_dim = 0
        self.output_dim = 0

        self.connected = False
        self.updates = []
        self.training_updates = []
        self.free_parameters = []
        self.parameters = []
        self.training_monitors = []
        self.testing_monitors = []
        self.external_inputs = []
        self.parameter_count = 0
        self.epoch_callbacks = []
        self.training_callbacks = []
        self.testing_callbacks = []


    def connect(self, input_dim, previous_layer=None, network_config=None):
        """
        Connect to a previous layer.
        """
        self.input_dim = input_dim
        self.previous_layer = previous_layer
        self.network_config = network_config
        self.connected = True

    def setup(self):
        """
        Setup function will be called after connected.
        """
        pass

    def output(self, x):
        """
        Output function.
        """
        raise NotImplementedError("output function of '%s' is not implemented" % self.name)

    def test_output(self, x):
        """
        Output function in test time.
        """
        return self.output(x)

    def register_parameters(self, *parameters):
        """
        Register parameters.
        """
        for param in parameters:
            self.parameter_count += np.prod(param.get_value().shape)
        self.parameters.extend(parameters)

    def register_free_parameters(self, *free_parameters):
        """
        Register free parameters, which means their value will not be learned by trainer.
        """
        return self.free_parameters.extend(free_parameters)

    def register_updates(self, *updates):
        """
        Register updates that will be executed in each iteration.
        """
        self.updates.extend(updates)

    def register_training_updates(self, *updates):
        """
        Register updates that will only be executed in training phase.
        """
        self.training_updates.extend(updates)

    def register_monitors(self, *monitors):
        """
        Register monitors they should be tuple of name and Theano variable.
        """
        self.training_monitors.extend(monitors)
        self.testing_monitors.extend(monitors)

    def register_external_inputs(self, *variables):
        """
        Register external input variables.
        """
        self.external_inputs.extend(variables)

    def register_training_callbacks(self, *callbacks):
        """
        Register callback for each iteration in the training.
        """
        self.training_callbacks.extend(callbacks)

    def register_testing_callbacks(self, *callbacks):
        """
        Register callback for each iteration in the testing.
        """
        self.testing_callbacks.extend(callbacks)

    def register_epoch_callbacks(self, *callbacks):
        """
        Register callback which will be called after epoch finished.
        """
        self.epoch_callbacks.extend(callbacks)

    def create_weight(self, input_n=1, output_n=1, suffix="", scheme=None, sparse=None, scale=None, shape=None):
        # ws = np.asarray(global_rand.uniform(low=-np.sqrt(6. / (input_n + output_n)),
        #                           high=np.sqrt(6. / (input_n + output_n)),
        #                           size=(input_n, output_n)))
        # if self.activation == 'sigmoid':
        #     ws *= 4
        # if sparse is not None:
        #     ws *= np.random.binomial(n=1, p=sparse, size=(input_n, output_n))

        adjust_weights = False
        if not shape:
            shape = (input_n, output_n)
            adjust_weights = True

        if not scale:
            scale = np.sqrt(6. / sum(shape))
            # if self.activation == 'sigmoid':
            #     scale *= 4

        ws = np.asarray(global_rand.uniform(low=-scale, high=scale, size=shape))

        # Adjust weights
        if adjust_weights:
            norm = np.sqrt((ws**2).sum())
            ws = scale * ws / norm
            _, v, _ = np.linalg.svd(ws)
            ws = scale * ws / v[0]

        weight = theano.shared(ws.astype(FLOATX), name='W_{}'.format(suffix))

        logging.info('create weight W_%s: %s', suffix, str(shape))
        return weight

    def create_bias(self, output_n=1, suffix="", value=0., shape=None):
        if not shape:
            shape = (output_n, )
        bs =  np.ones(shape)
        bs *= value
        bias = theano.shared(bs.astype(FLOATX), name='B_{}'.format(suffix))
        logging.info('create bias B_%s: %s', suffix, str(shape))
        return bias

    def create_vector(self, n, name, dtype=FLOATX):
        bs =  np.zeros(n)
        v = theano.shared(bs.astype(dtype), name='{}'.format(name))

        logging.info('create vector %s: %d', name, n)
        return v

    def create_matrix(self, m, n, name):

        matrix = theano.shared(np.zeros((m, n)).astype(FLOATX), name=name)

        logging.info('create matrix %s: %d x %d', name, m, n)
        return matrix

    def callback_forward_propagation(self):
        pass

