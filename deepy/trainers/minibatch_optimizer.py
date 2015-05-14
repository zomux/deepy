#!/usr/bin/env python
# -*- coding: utf-8 -*-


from collections import OrderedDict
import logging as loggers

import theano
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse

from deepy.utils.functions import FLOATX
from optimize import optimize_updates


logging = loggers.getLogger(__name__)

class MiniBatchOptimizer(object):

    def __init__(self, batch_size=32, realtime=True, clip=True):
        self.batch_size = batch_size
        self.batch_counter = theano.shared(np.cast[FLOATX](0), "batch_counter")
        self.updates = []
        self.caches = []
        self._compiled = False
        self.realtime = realtime
        self.clip = clip


    def setup(self, params, gparams, shapes=None, max_norm = 5.0, lr = 0.01, eps= 1e-6,
              rho=0.95, method="ADADELTA", beta=0.0, count=None, weight_l2 = 0):
        # Setup only once
        assert not self.updates

        if not shapes:
            shapes = params

        if not count:
            count = T.constant(1, dtype=FLOATX)
        else:
            count = T.cast(count, FLOATX)


        gcache   = [theano.shared(np.zeros_like(param.get_value(borrow=True), dtype=FLOATX),
                                  name="gcache_%s" % param.name)
                    for param in shapes]
        gcache_mean = [g / self.batch_counter for g in gcache]

        optimize_updates = optimize_updates(params, gcache_mean, shapes, max_norm, lr, eps, rho, method, beta,
                                               gsum_regularization=0.0001, weight_l2 = weight_l2, clip=self.clip)
        self.updates.extend(optimize_updates)
        self.caches.extend(gcache)

        if self.realtime:
            # Realtime update
            needs_update = self.batch_counter >= T.constant(self.batch_size)
            update_dict = OrderedDict()
            for param, update_val in optimize_updates:
                update_dict[param] = ifelse(needs_update, update_val, param)
            for cache, g in zip(gcache, gparams):
                update_dict[cache] = ifelse(needs_update, g, cache + g)
            update_dict[self.batch_counter] = ifelse(needs_update, count, self.batch_counter + count)
            return update_dict.items()

        else:
            # Manual update, perhaps in the end of one iteration
            gcache_updates = [(c, c+g) for c, g in zip(gcache, gparams)] + [(self.batch_counter, self.batch_counter + count)]
            return gcache_updates

    def _compile_update_func(self):
        logging.info("compile minibatch updates: %s" % " ".join([str(item[0]) for item in self.updates]))
        updates = (self.updates +
                   [(self.batch_counter, T.constant(0, dtype=FLOATX))] +
                   [(gc, gc * 0) for gc in self.caches])
        self._update_func = theano.function([], [], updates=updates)
        self._compiled = True

    def run(self):
        """
        Use this function to run optimization in non-realtime mode.
        :return:
        """
        if self.batch_counter.get_value(borrow=True) >= self.batch_size:
            self.force_optimize()

    def force_optimize(self):
        if not self._compiled:
            self._compile_update_func()
        self._update_func()
