#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import numpy as np
from deepy.utils import FLOATX

import logging as loggers
logging = loggers.getLogger(__name__)

class LearningRateAnnealer(object):
    """
    Learning rate annealer.
    """

    def __init__(self, trainer, patience=5, anneal_times=4):
        """
        :type trainer: deepy.trainers.trainers.NeuralTrainer
        """
        self._trainer = trainer
        self._iter = -1
        self._annealed_iter = -1
        self._patience = patience
        self._anneal_times = anneal_times
        self._annealed_times = 0
        self._learning_rate = self._trainer.config.learning_rate
        if type(self._learning_rate) == float:
            raise Exception("use LearningRateAnnealer.learning_rate to wrap the value in the config.")

    def invoke(self):
        """
        Run it, return whether to end training.
        """
        self._iter += 1
        if self._iter - max(self._trainer.best_iter, self._annealed_iter) >= self._patience:
            if self._annealed_times >= self._anneal_times:
                logging.info("ending")
                return True
            else:
                self._trainer.set_params(*self._trainer.best_params)
                self._learning_rate.set_value(self._learning_rate.get_value() * 0.5)
                self._annealed_times += 1
                self._annealed_iter = self._iter
                logging.info("annealed learning rate to %f" % self._learning_rate.get_value())
        return False

    @staticmethod
    def learning_rate(value=0.01):
        """
        Wrap learning rate.
        """
        return theano.shared(np.array(value, dtype=FLOATX), name="learning_rate")



