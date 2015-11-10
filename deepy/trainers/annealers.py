#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import numpy as np

from controllers import TrainingController
from deepy.utils import FLOATX, shared_scalar

import logging as loggers
logging = loggers.getLogger(__name__)

class LearningRateAnnealer(TrainingController):
    """
    Learning rate annealer.
    """

    def __init__(self, trainer, patience=3, anneal_times=4):
        """
        :type trainer: deepy.trainers.trainers.NeuralTrainer
        """
        super(LearningRateAnnealer, self).__init__(trainer)
        self._iter = -1
        self._annealed_iter = -1
        self._patience = patience
        self._anneal_times = anneal_times
        self._annealed_times = 0
        self._learning_rate = self._trainer.config.learning_rate
        if type(self._learning_rate) == float:
            raise Exception("use shared_scalar to wrap the value in the config.")

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
        return shared_scalar(value, name="learning_rate")


class ScheduledLearningRateAnnealer(TrainingController):
    """
    Anneal learning rate according to pre-scripted schedule.
    """

    def __init__(self, trainer, iter_start_halving=5, max_iters=10):
        super(ScheduledLearningRateAnnealer, self).__init__(trainer)
        logging.info("iteration to start halving learning rate: %d" % iter_start_halving)
        self.iter_start_halving = iter_start_halving
        self.max_iters = max_iters
        self._learning_rate = self._trainer.config.learning_rate
        self._iter = 0

    def invoke(self):
        self._iter += 1
        if self._iter >= self.iter_start_halving:
            self._trainer.set_params(*self._trainer.best_params)
            self._learning_rate.set_value(self._learning_rate.get_value() * 0.5)
            logging.info("halving learning rate to %f" % self._learning_rate.get_value())
        if self._iter >= self.max_iters - 1:
            logging.info("ending")
            return True
        return False


class ExponentialLearningRateAnnealer(TrainingController):
    """
    Exponentially decay learning rate after each update.
    """

    def __init__(self, trainer, decay_factor=1.000004, min_lr=.000001, debug=False):
        super(ExponentialLearningRateAnnealer, self).__init__(trainer)
        logging.info("exponentially decay learning rate with decay factor = %f" % decay_factor)
        self.decay_factor = np.array(decay_factor, dtype=FLOATX)
        self.min_lr = np.array(min_lr, dtype=FLOATX)
        self.debug = debug
        self._learning_rate = self._trainer.config.learning_rate
        if type(self._learning_rate) == float:
            raise Exception("use shared_scalar to wrap the value in the config.")
        self._trainer.network.training_callbacks.append(self.update_callback)

    def update_callback(self):
        if self._learning_rate.get_value() > self.min_lr:
            self._learning_rate.set_value(self._learning_rate.get_value() / self.decay_factor)

    def invoke(self):
        if self.debug:
            logging.info("learning rate: %.8f" % self._learning_rate.get_value())
        return False
