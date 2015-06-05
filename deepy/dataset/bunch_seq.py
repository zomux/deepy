#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging as loggers
logging = loggers.getLogger(__name__)

from . import MiniBatches
import numpy as np
from deepy.utils import global_rand, FakeGenerator

class BunchSequences(MiniBatches):
    """
    Arrange sequences in bunch mode.
    See http://mi.eng.cam.ac.uk/~xc257/papers/RNNLMTrain_Interspeech2014.pdf .
    """

    def __init__(self, dataset, batch_size=20, fragment_length=5):
        super(BunchSequences, self).__init__(dataset, batch_size=batch_size)
        self.fragment_length = fragment_length
        if self.fragment_length < 1:
            raise SystemError("fragment_length must be greater than 1")

    def _yield_data(self, subset):
        subset = list(subset)
        global_rand.shuffle(subset)

        bunch_stack_x = [[] for _ in range(self.size)]
        bunch_stack_y = [[] for _ in range(self.size)]

        for x, y in subset:
            stack_lens = map(len, bunch_stack_x)
            shortest_i = stack_lens.index(min(stack_lens))
            bunch_stack_x[shortest_i].extend(x)
            bunch_stack_y[shortest_i].extend(y)
        self._pad_zeros(bunch_stack_x)
        self._pad_zeros(bunch_stack_y)
        pieces_x = self._cut_to_pieces(bunch_stack_x)
        pieces_y = self._cut_to_pieces(bunch_stack_y)
        logging.info("%d pieces this time" % int(float(len(bunch_stack_x[0])) / self.fragment_length))
        for piece in zip(pieces_x, pieces_y):
            yield piece

    def _train_set(self):
        if not self.origin.train_set():
            return None
        return self._yield_data(self.origin.train_set())

    def train_set(self):
        return FakeGenerator(self, "_train_set")

    def train_size(self):
        size = len([_ for _ in self.train_set()])
        return size

    def _cut_to_pieces(self, bunch_stack):
        """
        :type bunch_stack: list of list of int
        """
        stack_len = len(bunch_stack[0])
        for i in xrange(0, stack_len, self.fragment_length):
            yield np.array(map(lambda stack: stack[i: i + self.fragment_length], bunch_stack))

    def _pad_zeros(self, bunch_stack):
        """
        :type bunch_stack: list of list
        """
        min_len = min(map(len, bunch_stack))
        for i in range(len(bunch_stack)):
            bunch_stack[i] = bunch_stack[i][:min_len]
        # for stack in bunch_stack:
        #     for _ in range(max_len - len(stack)):
        #         stack.append(0)