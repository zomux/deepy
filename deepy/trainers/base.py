#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
import numpy as np
import theano

from deepy.conf import TrainerConfig
from deepy.dataset import Dataset
from deepy.utils import Timer

from abc import ABCMeta, abstractmethod

from logging import getLogger
logging = getLogger("trainer")

class NeuralTrainer(object):
    """
    A base class for all trainers.
    """
    __metaclass__ = ABCMeta

    def __init__(self, network, config=None):
        """
        Basic neural network trainer.
        :type network: deepy.NeuralNetwork
        :type config: deepy.conf.TrainerConfig
        :return:
        """
        super(NeuralTrainer, self).__init__()

        self.config = None
        if isinstance(config, TrainerConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = TrainerConfig(config)
        else:
            self.config = TrainerConfig()
        # Model and network all refer to the computational graph
        self.model = self.network = network

        self.network.prepare_training()
        self._setup_costs()

        self.evaluation_func = None

        self.validation_frequency = self.config.validation_frequency
        self.min_improvement = self.config.min_improvement
        self.patience = self.config.patience
        self._iter_callbacks = []

        self.best_cost = 1e100
        self.best_epoch = 0
        self.best_params = self.copy_params()
        self._skip_batches = 0
        self._skip_epochs = 0
        self._progress = 0
        self.last_cost = 0
        self.last_run_costs = None
        self._report_time = True
        self._epoch = 0

    def _compile_evaluation_func(self):
        if not self.evaluation_func:
            logging.info("compile evaluation function")
            self.evaluation_func = theano.function(
                self.network.input_variables + self.network.target_variables,
                self.evaluation_variables,
                updates=self.network.updates,
                allow_input_downcast=True, mode=self.config.get("theano_mode", None))

    def skip(self, n_batches, n_epochs=0):
        """
        Skip N batches in the training.
        """
        logging.info("skip %d epochs and %d batches" % (n_epochs, n_batches))
        self._skip_batches = n_batches
        self._skip_epochs = n_epochs

    def epoch(self):
        """
        Get current epoch.
        """
        return self._epoch

    def _setup_costs(self):
        self.cost = self._add_regularization(self.network.cost)
        self.test_cost = self._add_regularization(self.network.test_cost)
        self.training_variables = [self.cost]
        self.training_names = ['J']
        for name, monitor in self.network.training_monitors:
            self.training_names.append(name)
            self.training_variables.append(monitor)
        logging.info("monitor list: %s" % ",".join(self.training_names))

        self.evaluation_variables = [self.test_cost]
        self.evaluation_names = ['J']
        for name, monitor in self.network.testing_monitors:
            self.evaluation_names.append(name)
            self.evaluation_variables.append(monitor)

    def _add_regularization(self, cost):
        if self.config.weight_l1 > 0:
            logging.info("L1 weight regularization: %f" % self.config.weight_l1)
            cost += self.config.weight_l1 * sum(abs(w).sum() for w in self.network.parameters)
        if self.config.hidden_l1 > 0:
            logging.info("L1 hidden unit regularization: %f" % self.config.hidden_l1)
            cost += self.config.hidden_l1 * sum(abs(h).mean(axis=0).sum() for h in self.network._hidden_outputs)
        if self.config.hidden_l2 > 0:
            logging.info("L2 hidden unit regularization: %f" % self.config.hidden_l2)
            cost += self.config.hidden_l2 * sum((h * h).mean(axis=0).sum() for h in self.network._hidden_outputs)

        return cost

    def set_params(self, targets, free_params=None):
        for param, target in zip(self.network.parameters, targets):
            param.set_value(target)
        if free_params:
            for param, param_value in zip(self.network.free_parameters, free_params):
                param.set_value(param_value)

    def save_params(self, path):
        self.set_params(*self.best_params)
        self.network.save_params(path)

    def load_params(self, path, exclude_free_params=False):
        """
        Load parameters for the training.
        This method can load free parameters and resume the training progress.
        """
        self.network.load_params(path, exclude_free_params=exclude_free_params)
        self.best_params = self.copy_params()
        # Resume the progress
        if self.network.train_logger.progress() > 0 or self.network.train_logger.epoch() > 0:
            self.skip(self.network.train_logger.progress(), self.network.train_logger.epoch() - 1)

    def copy_params(self):
        checkpoint = (map(lambda p: p.get_value().copy(), self.network.parameters),
                      map(lambda p: p.get_value().copy(), self.network.free_parameters))
        return checkpoint

    def add_iter_callback(self, func):
        """
        Add a iteration callback function (receives an argument of the trainer).
        :return:
        """
        self._iter_callbacks.append(func)

    def train(self, train_set, valid_set=None, test_set=None, train_size=None):
        """
        Train the model and return costs.
        """
        self._epoch = 0
        while True:
            if self._skip_epochs > 0:
                logging.info("skipping one epoch ...")
                self._skip_epochs -= 1
                self._epoch += 1
                yield None
                continue
            # Test
            if not self._epoch % self.config.test_frequency and test_set:
                try:
                    self._run_test(self._epoch, test_set)
                except KeyboardInterrupt:
                    logging.info('interrupted!')
                    break
            # Validate
            if not self._epoch % self.validation_frequency and valid_set:
                try:

                    if not self._run_valid(self._epoch, valid_set):
                        logging.info('patience elapsed, bailing out')
                        break
                except KeyboardInterrupt:
                    logging.info('interrupted!')
                    break
            # Train one step

            try:
                costs = self._run_train(self._epoch, train_set, train_size)
            except KeyboardInterrupt:
                logging.info('interrupted!')
                break
            # Check costs
            if np.isnan(costs[0][1]):
                logging.info("NaN detected in costs, rollback to last parameters")
                self.set_params(*self.checkpoint)
            else:
                self._epoch += 1
                self.network.epoch_callback()

            yield dict(costs)

        if valid_set and self.config.get("save_best_parameters", True):
            self.set_params(*self.best_params)
        if test_set:
            self._run_test(-1, test_set)

    @abstractmethod
    def learn(self, *variables):
        """
        Update the parameters and return the cost with given data points.
        :param variables:
        :return:
        """

    def _run_test(self, iteration, test_set):
        """
        Run on test iteration.
        """
        costs = self.test_step(test_set)
        info = ' '.join('%s=%.2f' % el for el in costs)
        message = "test    (epoch=%i) %s" % (iteration + 1, info)
        logging.info(message)
        self.network.train_logger.record(message)
        self.last_run_costs = costs

    def _run_train(self, epoch, train_set, train_size=None):
        """
        Run one training iteration.
        """
        self.network.train_logger.record_epoch(epoch + 1)
        costs = self.train_step(train_set, train_size)
        if not epoch % self.config.monitor_frequency:
            info = " ".join("%s=%.2f" % item for item in costs)
            message = "monitor (epoch=%i) %s" % (epoch + 1, info)
            logging.info(message)
            self.network.train_logger.record(message)
        self.last_run_costs = costs
        return costs

    def _run_valid(self, epoch, valid_set, dry_run=False):
        """
        Run one valid iteration, return true if to continue training.
        """
        costs = self.valid_step(valid_set)
        # this is the same as: (J_i - J_f) / J_i > min improvement
        _, J = costs[0]
        marker = ""
        if self.best_cost - J > self.best_cost * self.min_improvement:
            # save the best cost and parameters
            self.best_params = self.copy_params()
            marker = ' *'
            if not dry_run:
                self.best_cost = J
                self.best_epoch = epoch

            if self.config.auto_save and self._skip_batches == 0:
                self.network.train_logger.record_progress(self._progress)
                self.network.save_params(self.config.auto_save, new_thread=True)

        info = ' '.join('%s=%.2f' % el for el in costs)
        epoch_str = "epoch=%d" % (epoch + 1)
        if dry_run:
            epoch_str = "dryrun" + " " * (len(epoch_str) - 6)
        message = "valid   (%s) %s%s" % (epoch_str, info, marker)
        logging.info(message)
        self.last_run_costs = costs
        self.network.train_logger.record(message)
        self.checkpoint = self.copy_params()
        return epoch - self.best_epoch < self.patience

    def test_step(self, test_set):
        self._compile_evaluation_func()
        costs = list(zip(
            self.evaluation_names,
            np.mean([self.evaluation_func(*x) for x in test_set], axis=0)))
        return costs

    def valid_step(self, valid_set):
        self._compile_evaluation_func()
        costs = list(zip(
            self.evaluation_names,
            np.mean([self.evaluation_func(*x) for x in valid_set], axis=0)))
        return costs

    def train_step(self, train_set, train_size=None):
        dirty_trick_times = 0
        network_callback = bool(self.network.training_callbacks)
        trainer_callback = bool(self._iter_callbacks)
        cost_matrix = []
        exec_count = 0
        start_time = time.time()
        self._progress = 0


        for x in train_set:
            if self._skip_batches == 0:

                if dirty_trick_times > 0:
                    cost_x = self.learn(*[t[:(t.shape[0]/2)] for t in x])
                    cost_matrix.append(cost_x)
                    cost_x = self.learn(*[t[(t.shape[0]/2):] for t in x])
                    dirty_trick_times -= 1
                else:
                    try:
                        cost_x = self.learn(*x)
                    except MemoryError:
                        logging.info("Memory error was detected, perform dirty trick 30 times")
                        dirty_trick_times = 30
                        # Dirty trick
                        cost_x = self.learn(*[t[:(t.shape[0]/2)] for t in x])
                        cost_matrix.append(cost_x)
                        cost_x = self.learn(*[t[(t.shape[0]/2):] for t in x])
                cost_matrix.append(cost_x)
                self.last_cost = cost_x[0]
                exec_count += 1
                if network_callback:
                    self.network.training_callback()
                if trainer_callback:
                    for func in self._iter_callbacks:
                        func(self)
            else:
                self._skip_batches -= 1
            if train_size:
                self._progress += 1
                spd = float(exec_count) / (time.time() - start_time)
                sys.stdout.write("\x1b[2K\r> %d%% | J=%.2f | spd=%.2f batch/s" % (self._progress * 100 / train_size, self.last_cost, spd))
                sys.stdout.flush()
        self._progress = 0

        if train_size:
            sys.stdout.write("\r")
            sys.stdout.flush()
        costs = list(zip(self.training_names, np.mean(cost_matrix, axis=0)))
        return costs

    def run(self, train_set, valid_set=None, test_set=None, train_size=None, controllers=None):
        """
        Run until the end.
        """
        if isinstance(train_set, Dataset):
            dataset = train_set
            train_set = dataset.train_set()
            valid_set = dataset.valid_set()
            test_set = dataset.test_set()
            train_size = dataset.train_size()
        if controllers:
            for controller in controllers:
                controller.bind(self)
        timer = Timer()
        for _ in self.train(train_set, valid_set=valid_set, test_set=test_set, train_size=train_size):
            if controllers:
                ending = False
                for controller in controllers:
                    if hasattr(controller, 'invoke') and controller.invoke():
                        ending = True
                if ending:
                    break
        if self._report_time:
            timer.report()
