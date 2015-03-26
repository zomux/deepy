#!/usr/bin/env python
# -*- coding: utf-8 -*-



import logging as loggers
import gzip
import cPickle as pickle

import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse

from deepy.conf import TrainerConfig
from deepy.trainers.optimize import optimize_parameters
from deepy.trainers.util import wrap_core
from deepy.util.functions import FLOATX


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
                        self.network.updating_callback()
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


class SGDTrainer(NeuralTrainer):
    '''Stochastic gradient descent network trainer.'''

    def __init__(self, network, config=None):
        """
        Create a SGD trainer.
        :type network:
        :type config: deepy.conf.TrainerConfig
        :return:
        """
        super(SGDTrainer, self).__init__(network, config)

        self.momentum = self.config.momentum
        self.learning_rate = self.config.learning_rate

        logging.info('compiling %s learning function', self.__class__.__name__)

        network_updates = list(network.updates) + list(network.learning_updates)
        learning_updates = list(self.learning_updates())
        update_list = network_updates + learning_updates
        logging.info("network updates: %s" % " ".join(map(str, [x[0] for x in network_updates])))
        logging.info("learning updates: %s" % " ".join(map(str, [x[0] for x in learning_updates])))

        self.learning_func = theano.function(
            network.inputs,
            self.cost_exprs,
            updates=update_list, allow_input_downcast=True, mode=theano.Mode(linker=THEANO_LINKER))

    def learning_updates(self):
        for param in self.network.weights + self.network.biases:
            delta = self.learning_rate * T.grad(self.J, param)
            velocity = theano.shared(
                np.zeros_like(param.get_value()), name=param.name + '_vel')
            yield velocity, self.momentum * velocity - delta
            yield param, param + velocity

class PureSGDTrainer(NeuralTrainer):
    '''Stochastic gradient descent network trainer.'''

    def __init__(self, network, config=None):
        """
        Create a SGD trainer.
        :type network:
        :type config: deepy.conf.TrainerConfig
        :return:
        """
        super(PureSGDTrainer, self).__init__(network, config)

        self.learning_rate = self.config.learning_rate

        logging.info('compiling %s learning function', self.__class__.__name__)

        network_updates = list(network.updates) + list(network.learning_updates)
        learning_updates = list(self.learning_updates())
        update_list = network_updates + learning_updates
        logging.info("network updates: %s" % " ".join(map(str, [x[0] for x in network_updates])))
        logging.info("learning updates: %s" % " ".join(map(str, [x[0] for x in learning_updates])))

        self.learning_func = theano.function(
            network.inputs,
            self.cost_exprs,
            updates=update_list, allow_input_downcast=True, mode=theano.Mode(linker=THEANO_LINKER))

    def learning_updates(self):
        for param in self.network.weights + self.network.biases:
            delta = self.learning_rate * T.grad(self.J, param)
            yield param, param - delta


class BatchPureSGDTrainer(NeuralTrainer):
    """
    Batch SGD Trainer
    """

    def __init__(self, network, config=None, batch_size=20):
        """
        Create a SGD trainer.
        :type network:
        :type config: deepy.conf.TrainerConfig
        :return:
        """
        super(BatchPureSGDTrainer, self).__init__(network, config)

        self.learning_rate = self.config.learning_rate
        self.batch_size = batch_size

        logging.info('compiling %s learning function', self.__class__.__name__)

        network_updates = list(network.updates) + list(network.learning_updates)
        learning_updates = list(self.learning_updates())
        update_list = network_updates + learning_updates
        logging.info("network updates: %s" % " ".join(map(str, [x[0] for x in network_updates])))
        logging.info("learning updates: %s" % " ".join(map(str, [x[0] for x in learning_updates])))

        self.learning_func = theano.function(
            network.inputs,
            self.cost_exprs,
            updates=update_list, allow_input_downcast=True, mode=theano.Mode(linker=THEANO_LINKER))


    def learning_updates(self):
        batch_counter = theano.shared(np.array(0, dtype="int32"), "batch_counter")
        batch_size = self.batch_size
        to_update = batch_counter >= batch_size

        for param in self.network.weights + self.network.biases:
            # delta = self.learning_rate * T.grad(self.J, param)
            gsum = theano.shared(np.zeros(param.get_value().shape, dtype=FLOATX), "batch_gsum_%s" % param.name)
            yield gsum, ifelse(to_update, T.zeros_like(gsum), gsum + T.grad(self.J, param))
            delta = self.learning_rate * gsum / batch_size
            yield param, ifelse(to_update, param - delta, param)

        yield batch_counter, ifelse(to_update, T.constant(0, dtype="int32"), batch_counter + 1)


class AdaDeltaTrainer(NeuralTrainer):
    '''AdaDelta network trainer.'''

    def __init__(self, network, config=None):
        """
        Create a SGD trainer.
        :type network:
        :type config: deepy.conf.TrainerConfig
        :return:
        """
        super(AdaDeltaTrainer, self).__init__(network, config)


        logging.info('compiling %s learning function', self.__class__.__name__)

        network_updates = list(network.updates) + list(network.learning_updates)
        learning_updates = list(self.learning_updates())
        update_list = network_updates + learning_updates
        logging.info("network updates: %s" % " ".join(map(str, [x[0] for x in network_updates])))
        logging.info("learning updates: %s" % " ".join(map(str, [x[0] for x in learning_updates])))

        self.learning_func = theano.function(
            network.inputs,
            self.cost_exprs,
            updates=update_list, allow_input_downcast=True, mode=theano.Mode(linker=THEANO_LINKER))

    def learning_updates(self):
        params = self.network.weights + self.network.biases
        gparams = [T.grad(self.J, param) for param in params]
        return optimize_parameters(params, gparams, method="ADADELTA")


class AdaGradTrainer(NeuralTrainer):
    '''AdaDelta network trainer.'''

    def __init__(self, network, config=None, gsum_regularization=0.0001):
        """
        Create a SGD trainer.
        :type network:
        :type config: deepy.conf.TrainerConfig
        :return:
        """
        super(AdaGradTrainer, self).__init__(network, config)

        self.learning_rate = self.config.learning_rate
        self.gsum_regularization = gsum_regularization

        logging.info('compiling %s learning function', self.__class__.__name__)

        network_updates = list(network.updates) + list(network.learning_updates)
        learning_updates = list(self.learning_updates())
        update_list = network_updates + learning_updates
        logging.info("network updates: %s" % " ".join(map(str, [x[0] for x in network_updates])))
        logging.info("learning updates: %s" % " ".join(map(str, [x[0] for x in learning_updates])))

        self.learning_func = theano.function(
            network.inputs,
            self.cost_exprs,
            updates=update_list, allow_input_downcast=True, mode=theano.Mode(linker=THEANO_LINKER))

    def learning_updates(self):
        params = self.network.weights + self.network.biases
        gparams = [T.grad(self.J, param) for param in params]
        return optimize_parameters(params, gparams, method="ADAGRAD", lr=self.learning_rate, beta=self.config.update_l1, gsum_regularization=self.gsum_regularization)

class FineTuningAdaGradTrainer(AdaGradTrainer):
    '''AdaDelta network trainer.'''

    def __init__(self, network, config=None):
        """
        Create a SGD trainer.
        :type network:
        :type config: deepy.conf.TrainerConfig
        :return:
        """
        super(FineTuningAdaGradTrainer, self).__init__(network, config, 0)

class RmspropTrainer(SGDTrainer):
    """
    RmsProp
    """

    def learning_updates(self):
        for param in self.params:
            grad = T.grad(self.J, param)
            rms_ = theano.shared(
                np.zeros_like(param.get_value()), name=param.name + '_rms')
            rms = self.momentum * rms_ + (1 - self.momentum) * grad * grad
            yield rms_, rms
            yield param, param - self.learning_rate * grad / T.sqrt(rms + 1e-8)


class AdamTrainer(NeuralTrainer):
    """
    Optimization class of Adam.
    """

    def __init__(self, network, config=None):
        super(AdamTrainer, self).__init__(network, config)

        self.learning_rate = self.config.learning_rate

        logging.info('compiling %s learning function', self.__class__.__name__)

        network_updates = list(network.updates) + list(network.learning_updates)
        learning_updates = list(self.learning_updates())
        update_list = network_updates + learning_updates
        logging.info("network updates: %s" % " ".join(map(str, [x[0] for x in network_updates])))
        logging.info("learning updates: %s" % " ".join(map(str, [x[0] for x in learning_updates])))

        self.learning_func = theano.function(
            network.inputs,
            self.cost_exprs,
            updates=update_list, allow_input_downcast=True, mode=theano.Mode(linker=THEANO_LINKER))

    def learning_updates(self):
        from adam import adam_core
        params = self.network.weights + self.network.biases
        gradients = map(lambda p: T.grad(self.J, p), params)
        return wrap_core(adam_core, self.config, params, gradients)

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
            self.cost_exprs,
            updates=update_list, allow_input_downcast=True, mode=theano.Mode(linker=THEANO_LINKER))

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
        return self.ssgd2(self.J, self.network.weights + self.network.biases, learning_rate=self.learning_rate)

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
            self.cost_exprs,
            updates=update_list, allow_input_downcast=True, mode=theano.Mode(linker=THEANO_LINKER))