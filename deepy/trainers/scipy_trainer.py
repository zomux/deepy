#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging

from deepy.trainers.trainer import NeuralTrainer
import theano
import numpy as np
import theano.tensor as T
import scipy


class ScipyTrainer(NeuralTrainer):
    '''General trainer for neural nets using `scipy.optimize.minimize`.'''

    METHODS = ('bfgs', 'cg', 'dogleg', 'newton-cg', 'trust-ncg', 'l-bfgs-b')

    def flat_to_arrays(self, x):
        x = x.astype(self._dtype)
        return [x[o:o+n].reshape(s) for s, o, n in
                zip(self._shapes, self._starts, self._counts)]

    def arrays_to_flat(self, arrays):
        x = np.zeros((sum(self._counts), ), self._dtype)
        for arr, o, n in zip(arrays, self._starts, self._counts):
            x[o:o+n] = arr.ravel()
        return x

    def __init__(self, network, method, **kwargs):
        super(ScipyTrainer, self).__init__(network, **kwargs)

        self.method = method
        self.iterations = kwargs.get('num_updates', 100)

        logging.info('compiling gradient function')
        self.f_grad = theano.function(network.inputs, T.grad(self.J, self.params))

        self._shapes = [p.get_value(borrow=True).shape for p in self.params]
        self._counts = [np.prod(s) for s in self._shapes]
        self._starts = np.cumsum([0] + self._counts)[:-1]
        self._dtype = self.params[0].get_value().dtype


    def function_at(self, x, train_set):
        self.set_params(self.flat_to_arrays(x))
        return np.mean([self.evaluation_func(*x)[0] for x in train_set])

    def gradient_at(self, x, train_set):
        self.set_params(self.flat_to_arrays(x))
        grads = [[] for _ in range(len(self.params))]
        for x in train_set:
            for i, g in enumerate(self.f_grad(*x)):
                grads[i].append(np.asarray(g))
        return self.arrays_to_flat([np.mean(g, axis=0) for g in grads])

    def train(self, train_set, valid_set=None, **kwargs):
        def display(x):
            self.set_params(self.flat_to_arrays(x))
            costs = np.mean([self.evaluation_func(*x) for x in train_set], axis=0)
            cost_desc = ' '.join(
                '%s=%.2f' % el for el in zip(self.cost_names, costs))
            logging.info('scipy.%s %i %s', self.method, i + 1, cost_desc)

        for i in range(self.iterations):
            try:
                if not self.evaluate(i, valid_set):
                    logging.info('patience elapsed, bailing out')
                    break
            except KeyboardInterrupt:
                logging.info('interrupted!')
                break

            try:
                res = scipy.optimize.minimize(
                    fun=self.function_at,
                    jac=self.gradient_at,
                    x0=self.arrays_to_flat(self.best_params),
                    args=(train_set, ),
                    method=self.method,
                    callback=display,
                    options=dict(maxiter=self.validation_frequency),
                )
            except KeyboardInterrupt:
                logging.info('interrupted!')
                break

            self.set_params(self.flat_to_arrays(res.x))

            yield {'J': res.fun}

        self.set_params(self.best_params)