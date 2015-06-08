#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging as loggers
logging = loggers.getLogger(__name__)

from deepy.networks import NeuralNetwork
from deepy.layers import OneHotEmbedding
from deepy.utils import onehot, EPSILON
import theano.tensor as T

from cost import LMCost

class NeuralLM(NeuralNetwork):
    """
    LM Network.
    """

    def __init__(self, vocab_size, class_based=False, test_data=None, config=None):
        self.class_based = class_based
        super(NeuralLM, self).__init__(0, config, input_tensor=T.imatrix('x'))
        self.stack(OneHotEmbedding(vocab_size))
        self.test_data = test_data
        if test_data:
            self.epoch_callbacks.append(self._report_ppl)

    def setup_variables(self):
        super(NeuralLM, self).setup_variables()
        if not self.class_based:
            self.k = T.imatrix('k')
            self.target_variables.append(self.k)

    def _cost_func(self, y):
        if self.class_based:
            return y
        else:
            return LMCost(y, self.k).get()

    def _error_func(self, y):
        y2 = y.reshape((-1, y.shape[-1]))
        k2 = self.k.reshape((-1,))
        return 100 * T.mean(T.neq(T.argmax(y2[:k2.shape[0]], axis=1), k2))

    def _perplexity_func(self, y):
        return 2**self._cost_func(y)

    @property
    def cost(self):
        return self._cost_func(self.output)

    @property
    def test_cost(self):
        return self._cost_func(self.test_output)

    def predict(self, x):
        return self.compute(x).argmax(axis=1)

    def sample(self, input, steps):
        """
        Sample outputs from LM.
        """
        inputs = [[onehot(self.input_dim, x) for x in input]]
        for _ in range(steps):
            target = self.compute(inputs)[0,-1].argmax()
            input.append(target)
            inputs[0].append(onehot(self.input_dim, target))
        return input

    def prepare_training(self):
        if not self.class_based:
            self.training_monitors.append(("err", self._error_func(self.output)))
            self.testing_monitors.append(("err", self._error_func(self.test_output)))
        self.training_monitors.append(("approx_PPL", self._perplexity_func(self.output)))
        self.testing_monitors.append(("approx_PPL", self._perplexity_func(self.test_output)))
        super(NeuralLM, self).prepare_training()

    def _report_ppl(self):
        if not self.test_data: return
        import numpy as np
        costs = []
        train_set = list(self.test_data)
        for i in range(len(train_set)):
            k2 =train_set[i][1].reshape((-1,))
            y2 = self.compute(train_set[i][0]).reshape((-1, 10001))
            c = -np.log2(y2[np.arange(k2.shape[0]), k2]).mean()
            costs.append(c)
        print np.array(costs).mean()
        # return
        # logp_sum  = 0
        # word_count = 0
        # for x_batch, y_batch in self.test_data:
        #     yhat_batch = self.compute(x_batch)
        #     for x_seq, y_seq, yhat_seq in zip(x_batch, y_batch, yhat_batch):
        #         for x, y, yhat in  zip(x_seq, y_seq, yhat_seq):
        #             logp = - np.log2(yhat[y])
        #             logp_sum += logp
        #             word_count += 1
        #             if y == 0:
        #                 # End of the sentence
        #                 break
        # ppl = 0
        # mean_logp = 0
        # if word_count > 0:
        #     mean_logp = logp_sum / word_count
        #     ppl = 2 ** mean_logp
        # logging.info("Accurate test PPL: %.2f, Entropy: %.2f" % (ppl, mean_logp))