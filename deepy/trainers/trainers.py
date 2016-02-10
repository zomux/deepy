#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T

from deepy.conf import TrainerConfig
from deepy.trainers.base import NeuralTrainer
from deepy.trainers.optimize import optimize_updates

from logging import getLogger
logging = getLogger(__name__)

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

        self._learning_func = None

    def learn(self, *variables):
        if not self._learning_func:
            logging.info('compiling %s learning function', self.__class__.__name__)
            self._learning_func = self.learning_function()
        return self._learning_func(*variables)

    def _learning_updates(self):
        """
        Return updates in the training.
        """
        params = self.training_params()
        gradients = self.get_gradients(params)
        return self.optimization_updates(params, gradients)

    def training_params(self):
        """
        Get parameters to be optimized.
        """
        params = self.network.parameters
        # Freeze parameters
        if self.config.fixed_parameters:
            logging.info("fixed parameters: %s" % ", ".join(map(str, self.config.fixed_parameters)))
            params = [p for p in params if p not in self.config.fixed_parameters]
        return params

    def get_gradients(self, params):
        """
        Get gradients from given parameters.
        """
        return T.grad(self.cost, params)

    def optimization_updates(self, params, gradients):
        """
        Return updates from optimization.
        """
        updates, free_parameters = optimize_updates(params, gradients, self.config)
        self.network.free_parameters.extend(free_parameters)
        logging.info("Added %d free parameters for optimization" % len(free_parameters))
        return updates

    def learning_function(self):
        """
        Get the learning function.
        :param func:
        :return:
        """
        network_updates = list(self.network.updates) + list(self.network.training_updates)
        learning_updates = list(self._learning_updates())
        update_list = network_updates + learning_updates

        logging.info("network updates: %s" % " ".join(map(str, [x[0] for x in network_updates])))
        logging.info("learning updates: %s" % " ".join(map(str, [x[0] for x in learning_updates])))

        variables = self.network.input_variables + self.network.target_variables
        givens = None
        return theano.function(
            variables,
            map(lambda v: theano.Out(v, borrow=True), self.training_variables),
            updates=update_list, allow_input_downcast=True,
            mode=self.config.get("theano_mode", None),
            givens=givens)


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

class FakeTrainer(GeneralNeuralTrainer):
    """
    Fake Trainer does nothing.
    """

    def _learning_updates(self):
        return []