#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from controllers import TrainingController
from deepy.core.env import FLOATX
from deepy.core import graph

import logging as loggers
logging = loggers.getLogger(__name__)

class LearningRateAnnealer(TrainingController):
    """
    Learning rate annealer.
    """

    def __init__(self, patience=3, anneal_times=4, anneal_factor=0.5):
        """
        :type trainer: deepy.trainers.base.NeuralTrainer
        """
        self._iter = 0
        self._annealed_iter = 0
        self._patience = patience
        self._anneal_times = anneal_times
        self._anneal_factor = anneal_factor
        self._annealed_times = 0
        self._learning_rate = None

    def bind(self, trainer):
        super(LearningRateAnnealer, self).bind(trainer)
        self._learning_rate = self._trainer.config.learning_rate
        if type(self._learning_rate) == float:
            raise Exception("use shared_scalar to wrap the value in the config.")
        assert self._learning_rate
        self._iter = 0
        self._annealed_iter = 0

    def invoke(self):
        """
        Run it, return whether to end training.
        """
        self._iter += 1
        if self._iter - max(self._trainer.best_iter, self._annealed_iter) >= self._patience:
            if self._annealed_times >= self._anneal_times:
                logging.info("ending")
                self._trainer.exit()
            else:
                self._trainer.set_params(*self._trainer.best_params)
                self._learning_rate.set_value(
                    np.array(self._learning_rate.get_value() * self._anneal_factor, dtype=FLOATX))
                self._annealed_times += 1
                self._annealed_iter = self._iter
                logging.info("annealed learning rate to %f" % self._learning_rate.get_value())

    @staticmethod
    def learning_rate(value=0.01):
        """
        Wrap learning rate.
        """
        return graph.shared(value, name="learning_rate")


class ScheduledLearningRateAnnealer(TrainingController):
    """
    Anneal learning rate according to pre-scripted schedule.
    """

    def __init__(self, start_halving_at=5, end_at=10, halving_interval=1, rollback=False):
        logging.info("iteration to start halving learning rate: %d" % start_halving_at)
        self.epoch_start_halving = start_halving_at
        self.end_at = end_at
        self._halving_interval = halving_interval
        self._rollback = rollback
        self._last_halving_epoch = 0
        self._learning_rate = None

    def bind(self, trainer):
        super(ScheduledLearningRateAnnealer, self).bind(trainer)
        self._learning_rate = self._trainer.config.learning_rate
        self._last_halving_epoch = 0

    def invoke(self):
        epoch = self._trainer.epoch()
        if epoch >= self.epoch_start_halving and epoch >= self._last_halving_epoch + self._halving_interval:
            if self._rollback:
                self._trainer.set_params(*self._trainer.best_params)
            self._learning_rate.set_value(
                np.array(self._learning_rate.get_value() * 0.5, dtype=FLOATX))
            logging.info("halving learning rate to %f" % self._learning_rate.get_value())
            self._trainer.network.train_logger.record("set learning rate to %f" % self._learning_rate.get_value())
            self._last_halving_epoch = epoch
        if epoch >= self.end_at:
            logging.info("ending")
            self._trainer.exit()


class ExponentialLearningRateAnnealer(TrainingController):
    """
    Exponentially decay learning rate after each update.
    """

    def __init__(self, decay_factor=1.000004, min_lr=.000001, debug=False):
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


class SimpleScheduler(TrainingController):

    """
    Simple scheduler with maximum patience.
    """

    def __init__(self, end_at=10):
        """
        :type trainer: deepy.trainers.base.NeuralTrainer
        """
        self._iter = 0
        self._patience = end_at

    def invoke(self):
        """
        Run it, return whether to end training.
        """
        self._iter += 1
        logging.info("{} epochs left to run".format(self._patience - self._iter))
        if self._iter >= self._patience:
            self._trainer.exit()
