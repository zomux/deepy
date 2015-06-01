#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging as loggers
import gzip, sys
import cPickle as pickle

import numpy as np
import theano
import theano.tensor as T

from deepy.conf import TrainerConfig
from deepy.dataset import Dataset
from deepy.trainers.optimize import optimize_updates
from deepy.utils import Timer


logging = loggers.getLogger(__name__)

THEANO_LINKER = 'cvm'

def inspect_inputs(i, node, fn):
    print i, node, "input(s) value(s):", [input[0] for input in fn.inputs],

def inspect_outputs(i, node, fn):
    print "output(s) value(s):", [output[0] for output in fn.outputs]

def default_mapper(f, dataset, *args, **kwargs):
    '''Apply a function to each element of a dataset.'''
    return [f(x, *args, **kwargs) for x in dataset]

def ipcluster_mapper(client):
    '''Get a mapper from an IPython.parallel cluster client.'''
    view = client.load_balanced_view()
    def mapper(f, dataset, *args, **kwargs):
        def ff(x):
            return f(x, *args, **kwargs)
        return view.map(ff, dataset).get()
    return mapper


class NeuralTrainer(object):
    '''This is a base class for all trainers.'''

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
        self.network = network

        self.network.prepare_training()

        self._setup_costs()

        logging.info("compile evaluation function")
        self.evaluation_func = theano.function(
            network.input_variables + network.target_variables, self.evaluation_variables, updates=network.updates,
            allow_input_downcast=True, mode=self.config.get("theano_mode", None))
        self.learning_func = None

        self.validation_frequency = self.config.validation_frequency
        self.min_improvement = self.config.min_improvement
        self.patience = self.config.patience

        self.best_cost = 1e100
        self.best_iter = 0
        self.best_params = self._copy_network_params()

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

    def load_params(self, path):
        self.network.load_params(path)
        self.best_params = self._copy_network_params()

    def _copy_network_params(self):
        checkpoint = (map(lambda p: p.get_value().copy(), self.network.parameters),
                      map(lambda p: p.get_value().copy(), self.network.free_parameters))
        return checkpoint


    def train(self, train_set, valid_set=None, test_set=None, train_size=None):
        """
        Train the model and return costs.
        """
        if not self.learning_func:
            raise NotImplementedError
        iteration = 0
        while True:
            # Test
            if not iteration % self.config.test_frequency and test_set:
                try:
                    self._run_test(iteration, test_set)
                except KeyboardInterrupt:
                    logging.info('interrupted!')
                    break
            # Validate
            if not iteration % self.validation_frequency and valid_set:
                try:

                    if not self._run_valid(iteration, valid_set):
                        logging.info('patience elapsed, bailing out')
                        break
                except KeyboardInterrupt:
                    logging.info('interrupted!')
                    break
            # Train one step
            try:
                costs = self._run_train(iteration, train_set, train_size)
            except KeyboardInterrupt:
                logging.info('interrupted!')
                break
            # Check costs
            if np.isnan(costs[0][1]):
                logging.info("NaN detected in costs, rollback to last parameters")
                self.set_params(*self.checkpoint)
            else:
                iteration += 1
                self.network.epoch_callback()

            yield dict(costs)

        if valid_set and self.config.get("save_best_parameters", True):
            self.set_params(*self.best_params)
        if test_set:
            self._run_test(-1, test_set)

    def _run_test(self, iteration, test_set):
        """
        Run on test iteration.
        """
        costs = self.test_step(test_set)
        info = ' '.join('%s=%.2f' % el for el in costs)
        message = "test    (iter=%i) %s" % (iteration + 1, info)
        logging.info(message)
        self.network.train_logger.record(message)

    def _run_train(self, iteration, train_set, train_size=None):
        """
        Run one training iteration.
        """
        costs = self.train_step(train_set, train_size)

        if not iteration % self.config.monitor_frequency:
            info = " ".join("%s=%.2f" % item for item in costs)
            message = "monitor (iter=%i) %s" % (iteration + 1, info)
            logging.info(message)
            self.network.train_logger.record(message)
        return costs

    def _run_valid(self, iteration, valid_set):
        """
        Run one valid iteration, return true if to continue training.
        """
        costs = self.valid_step(valid_set)
        # this is the same as: (J_i - J_f) / J_i > min improvement
        _, J = costs[0]
        if self.best_cost - J > self.best_cost * self.min_improvement:
            self.best_cost = J
            self.best_iter = iteration
            self.best_params = self._copy_network_params()
            marker = ' *'
        else:
            marker = ""
        info = ' '.join('%s=%.2f' % el for el in costs)
        message = "valid   (iter=%i) %s%s" % (iteration + 1, info, marker)
        logging.info(message)
        self.network.train_logger.record(message)
        self.checkpoint = self._copy_network_params()
        return iteration - self.best_iter < self.patience

    def test_step(self, test_set):
        costs = list(zip(
            self.evaluation_names,
            np.mean([self.evaluation_func(*x) for x in test_set], axis=0)))
        return costs

    def valid_step(self, valid_set):
        costs = list(zip(
            self.evaluation_names,
            np.mean([self.evaluation_func(*x) for x in valid_set], axis=0)))
        return costs

    def train_step(self, train_set, train_size=None):
        training_callback = bool(self.network.training_callbacks)
        cost_matrix = []
        c = 0

        for x in train_set:
            cost_x = self.learning_func(*x)
            cost_matrix.append(cost_x)
            if training_callback:
                self.network.training_callback()
            if train_size:
                c += 1
                sys.stdout.write("\r> %d%%" % (c * 100 / train_size))
                sys.stdout.flush()

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

        timer = Timer()
        for _ in self.train(train_set, valid_set=valid_set, test_set=test_set, train_size=train_size):
            if controllers:
                ending = False
                for controller in controllers:
                    if hasattr(controller, 'invoke') and controller.invoke():
                        ending = True
                if ending:
                    break
        timer.report()
        return

class GeneralNeuralTrainer(NeuralTrainer):
    """
    General neural network trainer.
    """
    def __init__(self, network, config=None, method=None):

        if method:
            logging.info("changing optimization method to '%s'" % method)
            if not config:
                config = TrainerConfig()
            elif isinstance(config, dict):
                config = TrainerConfig(config)
            config.method = method

        super(GeneralNeuralTrainer, self).__init__(network, config)

        logging.info('compiling %s learning function', self.__class__.__name__)

        network_updates = list(network.updates) + list(network.training_updates)
        learning_updates = list(self.learning_updates())
        update_list = network_updates + learning_updates
        logging.info("network updates: %s" % " ".join(map(str, [x[0] for x in network_updates])))
        logging.info("learning updates: %s" % " ".join(map(str, [x[0] for x in learning_updates])))

        self.learning_func = theano.function(
            network.input_variables + network.target_variables,
            map(lambda v: theano.Out(v, borrow=True), self.training_variables),
            updates=update_list, allow_input_downcast=True,
            mode=self.config.get("theano_mode", None))

    def learning_updates(self):
        """
        Return updates in the training.
        """
        params = self.network.parameters
        gradients = T.grad(self.cost, params)
        updates, free_parameters = optimize_updates(params, gradients, self.config)
        self.network.free_parameters.extend(free_parameters)
        logging.info("Added %d free parameters for optimization" % len(free_parameters))
        return updates


class SGDTrainer(GeneralNeuralTrainer):
    """
    SGD trainer.
    """
    def __init__(self, network, config=None):
        super(SGDTrainer, self).__init__(network, config, "SGD")

class AdaDeltaTrainer(GeneralNeuralTrainer):
    """
    AdaDelta trainer.
    """
    def __init__(self, network, config=None):
        super(AdaDeltaTrainer, self).__init__(network, config, "ADADELTA")


class AdaGradTrainer(GeneralNeuralTrainer):
    """
    AdaGrad trainer.
    """
    def __init__(self, network, config=None):
        super(AdaGradTrainer, self).__init__(network, config, "ADAGRAD")

class FineTuningAdaGradTrainer(GeneralNeuralTrainer):
    """
    AdaGrad trainer.
    """
    def __init__(self, network, config=None):
        super(FineTuningAdaGradTrainer, self).__init__(network, config, "FINETUNING_ADAGRAD")

class AdamTrainer(GeneralNeuralTrainer):
    """
    AdaGrad trainer.
    """
    def __init__(self, network, config=None):
        super(AdamTrainer, self).__init__(network, config, "ADAM")

class RmspropTrainer(GeneralNeuralTrainer):
    """
    RmsProp trainer.
    """
    def __init__(self, network, config=None):
        super(RmspropTrainer, self).__init__(network, config, "RMSPROP")

class MomentumTrainer(GeneralNeuralTrainer):
    """
    Momentum trainer.
    """
    def __init__(self, network, config=None):
        super(MomentumTrainer, self).__init__(network, config, "MOMENTUM")


class SSGD2Trainer(NeuralTrainer):
    """
    Optimization class of SSGD.
    """

    def __init__(self, network, config=None):
        super(SSGD2Trainer, self).__init__(network, config)

        self.learning_rate = self.config.learning_rate

        logging.info('compiling %s learning function', self.__class__.__name__)

        network_updates = list(network.updates) + list(network.learning_updates)
        learning_updates = list(self.learning_updates())
        update_list = network_updates + learning_updates
        logging.info("network updates: %s" % " ".join(map(str, [x[0] for x in network_updates])))
        logging.info("learning updates: %s" % " ".join(map(str, [x[0] for x in learning_updates])))

        self.learning_func = theano.function(
            network.inputs,
            self.training_variables,
            updates=update_list, allow_input_downcast=True, mode=self.config.get("theano_mode", None))

    def ssgd2(self, loss, all_params, learning_rate=0.01, chaos_energy=0.01, alpha=0.9):
        from theano.tensor.shared_randomstreams import RandomStreams

        chaos_energy = T.constant(chaos_energy, dtype="float32")
        alpha = T.constant(alpha, dtype="float32")
        learning_rate = T.constant(learning_rate, dtype="float32")

        srng = RandomStreams(seed=3)
        updates = []
        all_grads = T.grad(loss, all_params)
        for p, g in zip(all_params, all_grads):
            rand_v = (srng.uniform(p.get_value().shape)*2 - 1) * chaos_energy
            g_ratio_vec = g / g.norm(L=2)
            ratio_sum = theano.shared(np.ones(np.array(p.get_value().shape), dtype="float32"), name="ssgd2_r_sum_%s" % p.name)
            abs_ratio_sum = T.abs_(ratio_sum)
            updates.append((ratio_sum, ratio_sum * alpha + (1 - alpha ) * g_ratio_vec))
            updates.append((p, p - learning_rate*((abs_ratio_sum)*g + (1-abs_ratio_sum)*rand_v)))
        return updates

    def learning_updates(self):
        return self.ssgd2(self.cost, self.network.parameters, learning_rate=self.learning_rate)

class FakeTrainer(NeuralTrainer):
    """
    Fake Trainer does nothing.
    """

    def __init__(self, network, config=None):
        super(FakeTrainer, self).__init__(network, config)

        self.learning_rate = self.config.learning_rate

        logging.info('compiling %s learning function', self.__class__.__name__)

        network_updates = list(network.updates) + list(network.learning_updates)
        learning_updates = []
        update_list = network_updates + learning_updates
        logging.info("network updates: %s" % " ".join(map(str, [x[0] for x in network_updates])))
        logging.info("learning updates: %s" % " ".join(map(str, [x[0] for x in learning_updates])))

        self.learning_func = theano.function(
            network.inputs,
            self.training_variables,
            updates=update_list, allow_input_downcast=True, mode=self.config.get("theano_mode", None))
