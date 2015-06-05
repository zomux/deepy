#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import MiniBatches
import numpy as np

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
        subset.sort(key=lambda x: len(x[0]), reverse=True)

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
        return zip(pieces_x, pieces_y)

    def train_size(self):
        return len(self.train_set())

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
        max_len = max(map(len, bunch_stack))
        for stack in bunch_stack:
            for _ in range(max_len - len(stack)):
                stack.append(0)