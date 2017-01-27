#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
import numpy as np
import theano

from ..conf import TrainerConfig
from ..core import env, runtime
from ..utils import Timer
from ..dataset import Dataset
from controllers import TrainingController

from abc import ABCMeta, abstractmethod

from logging import getLogger
logging = getLogger("trainer")

class NeuralTrainer(object):
    """
    A base class for all trainers.
    """
    __metaclass__ = ABCMeta

    def __init__(self, network, config=None, validator=None, annealer=None):
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
        if type(self.config.learning_rate) == float:
            self.config.learning_rate = theano.shared(np.array(self.config.learning_rate, dtype=env.FLOATX), "lr")
        # Model and network all refer to the computational graph
        self.model = self.network = network

        self.network.prepare_training()
        self._setup_costs()

        self.evaluation_func = None

        self.validation_frequency = self.config.validation_frequency
        self.min_improvement = self.config.min_improvement
        self.patience = self.config.patience

        self._iter_controllers = []
        self._epoch_controllers = []
        if annealer:
            annealer.bind(self)
            self._epoch_controllers.append(annealer)
        if validator:
            validator.bind(self)
            self._iter_controllers.append(validator)

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

        self._current_train_set = None
        self._current_valid_set = None
        self._current_test_set = None
        self._ended = False


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

    def exit(self):
        """
        End the training.
        """
        self._ended = True

    def add_iter_controllers(self, *controllers):
        """
        Add iteration callbacks function (receives an argument of the trainer).
        :param controllers: can be a `TrainingController` or a function.
        :type funcs: list of TrainingContoller
        """
        for controller in controllers:
            if isinstance(controller, TrainingController):
                controller.bind(self)
            self._iter_controllers.append(controller)

    def add_epoch_controllers(self, *controllers):
        """
        Add epoch callbacks function.
        :param controllers: can be a `TrainingController` or a function.
        """
        for controller in controllers:
            if isinstance(controller, TrainingController):
                controller.bind(self)
            self._epoch_controllers.append(controller)

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

    def _run_test(self, epoch, test_set):
        """
        Run on test epoch.
        """
        costs = self.test_step(test_set)
        self.report(dict(costs), "test", epoch)
        self.last_run_costs = costs

    def _run_train(self, epoch, train_set, train_size=None):
        """
        Run one training iteration.
        """
        self.network.train_logger.record_epoch(epoch + 1)
        costs = self.train_step(train_set, train_size)
        if not epoch % self.config.monitor_frequency:
            self.report(dict(costs), "train", epoch)
        self.last_run_costs = costs
        return costs

    def _run_valid(self, epoch, valid_set, dry_run=False, save_path=None):
        """
        Run one valid iteration, return true if to continue training.
        """
        costs = self.valid_step(valid_set)
        # this is the same as: (J_i - J_f) / J_i > min improvement
        _, J = costs[0]
        new_best = False
        if self.best_cost - J > self.best_cost * self.min_improvement:
            # save the best cost and parameters
            self.best_params = self.copy_params()
            new_best = True
            if not dry_run:
                self.best_cost = J
                self.best_epoch = epoch
            self.save_checkpoint(save_path)

        self.report(dict(costs), type="valid", epoch=0 if dry_run else epoch, new_best=new_best)
        self.last_run_costs = costs
        return epoch - self.best_epoch < self.patience

    def save_checkpoint(self, save_path=None):
        save_path = save_path if save_path else self.config.auto_save
        self.checkpoint = self.copy_params()
        if save_path and self._skip_batches == 0:
            self.network.train_logger.record_progress(self._progress)
            self.network.save_params(save_path, new_thread=True)

    def report(self, score_map, type="valid", epoch=-1, new_best=False):
        """
        Report the scores and record them in the log.
        """
        type_str = type
        if len(type_str) < 5:
            type_str += " " * (5 - len(type_str))
        info = " ".join("%s=%.2f" % el for el in score_map.items())
        current_epoch = epoch if epoch > 0 else self.current_epoch()
        epoch_str = "epoch={}".format(current_epoch + 1)
        if epoch < 0:
            epoch_str = "dryrun"
            sys.stdout.write("\r")
            sys.stdout.flush()
        marker = " *" if new_best else ""
        message = "{} ({}) {}{}".format(type_str, epoch_str, info, marker)
        self.network.train_logger.record(message)
        logging.info(message)

    def test_step(self, test_set):
        runtime.switch_training(False)
        self._compile_evaluation_func()
        costs = list(zip(
            self.evaluation_names,
            np.mean([self.evaluation_func(*x) for x in test_set], axis=0)))
        return costs

    def valid_step(self, valid_set):
        runtime.switch_training(False)
        self._compile_evaluation_func()
        costs = list(zip(
            self.evaluation_names,
            np.mean([self.evaluation_func(*x) for x in valid_set], axis=0)))
        return costs

    def train_step(self, train_set, train_size=None):
        dirty_trick_times = 0
        network_callback = bool(self.network.training_callbacks)
        trainer_callback = bool(self._iter_controllers)
        cost_matrix = []
        exec_count = 0
        start_time = time.time()
        self._compile_time = 0
        self._progress = 0

        for x in train_set:
            runtime.switch_training(True)
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
                    for func in self._iter_controllers:
                        if isinstance(func, TrainingController):
                            func.invoke()
                        else:
                            func(self)
            else:
                self._skip_batches -= 1
            if train_size:
                self._progress += 1
                spd = float(exec_count) / (time.time() - start_time - self._compile_time)
                sys.stdout.write("\x1b[2K\r> %d%% | J=%.2f | spd=%.2f batch/s" % (self._progress * 100 / train_size, self.last_cost, spd))
                sys.stdout.flush()
        self._progress = 0

        if train_size:
            sys.stdout.write("\r")
            sys.stdout.flush()
        costs = list(zip(self.training_names, np.mean(cost_matrix, axis=0)))
        return costs

    def current_epoch(self):
        """
        Get current epoch.
        """
        return self._epoch


    def get_data(self, data_split="train"):
        """
        Get specified split of data.
        """
        if data_split == 'train':
            return self._current_train_set
        elif data_split == 'valid':
            return self._current_valid_set
        elif data_split == 'test':
            return self._current_test_set
        else:
            return None

    def run(self, train_set, valid_set=None, test_set=None, train_size=None, epoch_controllers=None):
        """
        Run until the end.
        :param epoch_controllers: deprecated
        """
        epoch_controllers = epoch_controllers if epoch_controllers else []
        epoch_controllers += self._epoch_controllers
        if isinstance(train_set, Dataset):
            dataset = train_set
            train_set = dataset.train_set()
            valid_set = dataset.valid_set()
            test_set = dataset.test_set()
            train_size = dataset.train_size()
        self._current_train_set = train_set
        self._current_valid_set = valid_set
        self._current_test_set = test_set
        if epoch_controllers:
            for controller in epoch_controllers:
                controller.bind(self)
        timer = Timer()
        for _ in self.train(train_set, valid_set=valid_set, test_set=test_set, train_size=train_size):
            if epoch_controllers:
                for controller in epoch_controllers:
                    controller.invoke()
            if self._ended:
                break
        if self._report_time:
            timer.report()
