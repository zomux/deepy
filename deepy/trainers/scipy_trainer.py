#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging

from deepy.trainers import NeuralTrainer, THEANO_LINKER
from deepy.utils import FLOATX
import theano
import numpy as np
import theano.tensor as T
import scipy


class ScipyTrainer(NeuralTrainer):
    """
    Optimizer based on Scipy.
    This class was modified based on the corresponding one the theanets.
    """

    METHODS = ('bfgs', 'cg', 'dogleg', 'newton-cg', 'trust-ncg', 'l-bfgs-b')

    def __init__(self, network, method, config=None):
        super(ScipyTrainer, self).__init__(network, config)

        self.method = method
        # Updates in one iteration
        self.scipy_updates = config.get("scipy_updates", 5) if config else 5

        logging.info('compiling gradient function')
        self._shapes = [p.get_value(borrow=True).shape for p in self.network.parameters]
        self._counts = [np.prod(s) for s in self._shapes]
        self._starts = np.cumsum([0] + self._counts)[:-1]
        self._dtype = FLOATX
        self._gradient_func = None
        # Declares that the learning function is implemented
        self.learning_func = True

    def train_step(self, train_set, train_size=None):

        res = scipy.optimize.minimize(
            fun=self._function_at,
            jac=self._gradient_at,
            x0=self._arrays_to_flat(self.best_params[0]),
            args=(train_set, ),
            method=self.method,
            options=dict(maxiter=self.scipy_updates),
        )

        self.set_params(self._flat_to_arrays(res.x))

        return [('J', res.fun)]

    def _gradient_function(self):
        if not self._gradient_func:
            params = self.network.parameters
            inputs = self.network.input_variables + self.network.target_variables
            self._gradient_func = theano.function(inputs, T.grad(self.cost, params),
                                        allow_input_downcast=True, mode=theano.Mode(linker=THEANO_LINKER))
        return self._gradient_func

    def _function_at(self, x, train_set):
        self.set_params(self._flat_to_arrays(x))
        return np.mean([self.evaluation_func(*x)[0] for x in train_set])

    def _gradient_at(self, x, train_set):
        self.set_params(self._flat_to_arrays(x))
        grads = [[] for _ in range(len(self.network.parameters))]
        grad_func = self._gradient_function()
        for x in train_set:
            for i, g in enumerate(grad_func(*x)):
                grads[i].append(np.asarray(g))
        return self._arrays_to_flat([np.mean(g, axis=0) for g in grads])

    def _flat_to_arrays(self, x):
        x = x.astype(self._dtype)
        return [x[o:o+n].reshape(s) for s, o, n in
                zip(self._shapes, self._starts, self._counts)]

    def _arrays_to_flat(self, arrays):
        x = np.zeros((sum(self._counts), ), self._dtype)
        for arr, o, n in zip(arrays, self._starts, self._counts):
            x[o:o+n] = arr.ravel()
        return x