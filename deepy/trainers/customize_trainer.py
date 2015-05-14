#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging as loggers
from abc import ABCMeta, abstractmethod

from deepy.trainers import NeuralTrainer


logging = loggers.getLogger(__name__)
THEANO_LINKER = 'cvm'

class CustomizeTrainer(NeuralTrainer):
    '''This is a base class for all trainers.'''
    __metaclass__ = ABCMeta

    def __init__(self, network, config=None):
        """
        Basic neural network trainer.
        :type network: deepy.NeuralNetwork
        :type config: deepy.conf.TrainerConfig
        :return:
        """
        super(CustomizeTrainer, self).__init__(network, config)


    def train(self, train_set, valid_set=None, test_set=None, train_size=None):
        '''We train over mini-batches and evaluate periodically.'''
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

            train_message = ""
            try:
                train_message = self.train_func(train_set)
            except KeyboardInterrupt:
                logging.info('interrupted!')
                break
            if not iteration % self.config.monitor_frequency:
                logging.info('monitor (iter=%i) %s', iteration + 1, train_message)

            iteration += 1
            if hasattr(self.network, "iteration_callback"):
                self.network.iteration_callback()

            yield train_message

        if valid_set:
            self.set_params(self.best_params)
        if test_set:
            self.test(0, test_set)

    @abstractmethod
    def train_func(self, train_set):
        return ""
    