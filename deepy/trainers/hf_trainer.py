#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging as loggers
import gzip
import cPickle as pickle

import numpy as np
import theano
from deepy.conf import TrainerConfig
from trainer import NeuralTrainer


logging = loggers.getLogger(__name__)

THEANO_LINKER = 'cvm'

class HFTrainer(NeuralTrainer):

    def __init__(self, network, config=None):
        """
        Basic neural network trainer.
        :type network: deepy.NeuralNetwork
        :type config: deepy.conf.TrainerConfig
        :return:
        """
        super(NeuralTrainer, self).__init__()

        self.params = network.params
        self.config = config if config else TrainerConfig()
        self.network = network

        self.J = network.J(self.config)
        self.cost_exprs = [self.J]
        self.cost_names = ['J']
        for name, monitor in network.monitors:
            self.cost_names.append(name)
            self.cost_exprs.append(monitor)
        logging.info("monitor list: %s" % ",".join(self.cost_names))


        logging.info('compiling evaluation function')
        self.ev_cost_exprs = []
        self.ev_cost_names = []
        for i in range(len(self.cost_names)):
            if self.cost_names[i].endswith("x"):
                continue
            self.ev_cost_exprs.append(self.cost_exprs[i])
            self.ev_cost_names.append(self.cost_names[i])
        self.evaluation_func = theano.function(
            network.inputs, self.ev_cost_exprs, updates=network.updates, allow_input_downcast=True,
            mode=theano.Mode(linker=THEANO_LINKER))
            # mode=theano.compile.MonitorMode(
            #             pre_func=inspect_inputs,
            #             post_func=inspect_outputs) )

        self.validation_frequency = self.config.validation_frequency
        self.min_improvement = self.config.min_improvement
        self.patience = self.config.patience

        self.shapes = [p.get_value(borrow=True).shape for p in self.params]
        self.counts = [np.prod(s) for s in self.shapes]
        self.starts = np.cumsum([0] + self.counts)[:-1]

        self.best_cost = 1e100
        self.best_iter = 0
        self.best_params = [p.get_value().copy() for p in self.params]

    def set_params(self, targets):
        for param, target in zip(self.params, targets):
            param.set_value(target)

    def test(self, iteration, test_set):
        costs = list(zip(
            self.ev_cost_names,
            np.mean([self.evaluation_func(*x) for x in test_set], axis=0)))
        info = ' '.join('%s=%.2f' % el for el in costs)
        logging.info('test    (iter=%i) %s', iteration + 1, info)

    def evaluate(self, iteration, valid_set):
        costs = list(zip(
            self.ev_cost_names,
            np.mean([self.evaluation_func(*x) for x in valid_set], axis=0)))
        marker = ''
        # this is the same as: (J_i - J_f) / J_i > min improvement
        _, J = costs[0]
        if self.best_cost - J > self.best_cost * self.min_improvement:
            self.best_cost = J
            self.best_iter = iteration
            self.best_params = [p.get_value().copy() for p in self.params]
            marker = ' *'
        info = ' '.join('%s=%.2f' % el for el in costs)
        logging.info('valid   (iter=%i) %s%s', iteration + 1, info, marker)
        return iteration - self.best_iter < self.patience

    def save_params(self, path):
        logging.info("saving parameters to %s" % path)
        opener = gzip.open if path.lower().endswith('.gz') else open
        handle = opener(path, 'wb')
        pickle.dump(self.best_params, handle)
        handle.close()

    def train(self, train_set, valid_set=None, test_set=None):
        '''We train over mini-batches and evaluate periodically.'''
        if not hasattr(self, 'learning_func'):
            raise NotImplementedError
        iteration = 0
        while True:
            if not iteration % self.config.test_frequency and test_set:
                try:
                    self.test(iteration, test_set)
                except KeyboardInterrupt:
                    logging.info('interrupted!')
                    break

            if not iteration % self.validation_frequency and valid_set:
                try:
                    if not self.evaluate(iteration, valid_set):
                        logging.info('patience elapsed, bailing out')
                        break
                except KeyboardInterrupt:
                    logging.info('interrupted!')
                    break

            try:
                cost_matrix = []
                for x in train_set:
                    cost_matrix.append(self.learning_func(*x))
                    if self.network.needs_callback:
                        self.network.training_callback()
                costs = list(zip(self.cost_names, np.mean(cost_matrix, axis=0)))
            except KeyboardInterrupt:
                logging.info('interrupted!')
                break
            if not iteration % self.config.monitor_frequency:
                info = ' '.join('%s=%.2f' % el for el in costs)
                logging.info('monitor (iter=%i) %s', iteration + 1, info)

            iteration += 1
            if hasattr(self.network, "iteration_callback"):
                self.network.iteration_callback()

            yield dict(costs)

        if valid_set:
            self.set_params(self.best_params)
        if test_set:
            self.test(0, test_set)