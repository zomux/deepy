#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Some codes in this file are refactored from theanonets

import logging as loggers
import gzip
import cPickle as pickle

import theano.tensor as T
import theano


logging = loggers.getLogger(__name__)

from deepy.util.functions import VarMap
from deepy.util import build_activation, add_noise


class NeuralNetwork(object):

    def __init__(self, config):
        """
        :type config: deepy.conf.NetworkConfig
        :return:
        """
        self.config = config
        self.vars = VarMap()
        self.hiddens = []
        self.weights = []
        self.biases = []
        self.learning_updates = []
        self.updates = []
        self.inputs = []
        self.target_inputs = []
        self.special_params = []
        self.special_monitors = []
        self.updating_callbacks = []
        self.iteration_callbacks = []


        self.layers = config.layers

        self.setup_vars()
        self.vars.y, count = self.setup_layers()

        logging.info("total network parameters: %d", count)
        logging.info("network inputs: %s", " ".join(map(str, self.inputs)))
        logging.info("network params: %s", " ".join(map(str, self.params)))

    def updating_callback(self):
        for cb in self.updating_callbacks:
            cb()

    def setup_vars(self):
        self.vars.x = T.matrix('x')
        self.inputs.append(self.vars.x)


    def setup_layers(self):
        last_size = self.config.input_size
        parameter_count = 0
        z = add_noise(
            self.vars.x,
            self.config.input_noise,
            self.config.input_dropouts)
        for i, layer in enumerate(self.layers):
            size = layer.size
            layer.connect(self.config, self.vars, z, last_size, i + 1)
            parameter_count += layer.param_count
            self.hiddens.append(layer.output_func)
            if type(layer.W) == list:
                self.weights.extend(layer.W)
            else:
                self.weights.append(layer.W)
            if type(layer.B) == list:
                self.biases.extend(layer.B)
            else:
                self.biases.append(layer.B)
            self.special_params.extend(layer.params)
            self.special_monitors.extend(layer.monitors)
            self.updates.extend(layer.updates)
            self.learning_updates.extend(layer.learning_updates)
            self.inputs.extend(layer.inputs)
            if 'updating_callback' in dir(layer):
                self.updating_callbacks.append(layer.updating_callback)
            if 'iteration_callback' in dir(layer):
                self.iteration_callbacks.append(layer.iteration_callback)
            z = layer.output_func
            last_size = size
        self.needs_callback = bool(self.updating_callbacks)
        return self.hiddens.pop(), parameter_count

    @property
    def monitors(self):
        '''Generate a sequence of name-value pairs for monitoring the network.
        '''
        yield 'err', self.cost
        for i, h in enumerate(self.hiddens):
            yield 'h{}<0.1'.format(i+1), 100 * (abs(h) < 0.1).mean()
            yield 'h{}<0.9'.format(i+1), 100 * (abs(h) < 0.9).mean()
        for name, exp in self.special_monitors:
            yield name, exp

    def _compile(self):
        if getattr(self, '_compute', None) is None:
            self._compute = theano.function(
                [x for x in self.inputs if x not in self.target_inputs],
                self.hiddens + [self.vars.y], updates=self.updates, allow_input_downcast=True)

    @property
    def params(self):
        '''Return a list of the Theano parameters for this network.'''
        params = []
        params.extend(self.weights)
        params.extend(self.biases)
        params.extend(self.special_params)

        return params

    def set_params(self, params):
        self.weights, self.biases = params

    def feed_forward(self, *x):
        self._compile()
        return self._compute(*x)

    def predict(self, *x):
        return self.feed_forward(*x)[-1]

    __call__ = predict


    def J(self, train_conf):
        cost = self.cost

        if train_conf.weight_l1 > 0:
            logging.info("L1 weight regularization: %f" % train_conf.weight_l2)
            cost += train_conf.weight_l1 * sum(abs(w).sum() for w in self.weights)
        if train_conf.hidden_l1 > 0:
            logging.info("L1 hidden unit regularization: %f" % train_conf.weight_l2)
            cost += train_conf.hidden_l1 * sum(abs(h).mean(axis=0).sum() for h in self.hiddens)

        if train_conf.weight_l2 > 0:
            logging.info("L2 weight regularization: %f" % train_conf.weight_l2)
            cost += train_conf.weight_l2 * sum((w * w).sum() for w in self.weights)
        if train_conf.hidden_l2 > 0:
            logging.info("L2 hidden unit regularization: %f" % train_conf.hidden_l2)
            cost += train_conf.hidden_l2 * sum((h * h).mean(axis=0).sum() for h in self.hiddens)
        if train_conf.contractive_l2 > 0:
            logging.info("L2 contractive regularization: %f" % train_conf.hidden_l2)
            cost += train_conf.contractive_l2 * sum(
                T.sqr(T.grad(h.mean(axis=0).sum(), self.vars.x)).sum() for h in self.hiddens)

        return cost

    def save_params(self, path):
        logging.info("saving parameters to %s" % path)
        opener = gzip.open if path.lower().endswith('.gz') else open
        handle = opener(path, 'wb')
        pickle.dump([p.get_value().copy() for p in self.params], handle)
        handle.close()

    def load_params(self, path):
        logging.info("loading parameters from %s" % path)
        opener = gzip.open if path.lower().endswith('.gz') else open
        handle = opener(path, 'rb')
        saved = pickle.load(handle)
        for target, source in zip(self.params, saved):
            logging.info('%s: setting value %s', target.name, source.shape)
            target.set_value(source)
        handle.close()

    def iteration_callback(self):
        for cb in self.iteration_callbacks:
            cb()