#!/usr/bin/env python
# -*- coding: utf-8 -*-


from . import Dataset
from deepy.util import FLOATX
import numpy as np

class MiniBatches(Dataset):

    def __init__(self, dataset, batch_size=20):
        self.origin = dataset
        self.size = batch_size

    def _yield_data(self, subset):
        for i in xrange(0, len(subset), self.size):
            x_set, y_set = [], []
            for x, y in subset[i:i + self.size]:
                x_set.append(x)
                y_set.append(y)
            x_set = np.array(x_set)
            y_set = np.array(y_set)
            yield x_set, y_set

    def train_set(self):
        return list(self._yield_data(self.origin.train_set()))

    def test_set(self):
        return list(self._yield_data(self.origin.test_set()))

    def valid_set(self):
        return list(self._yield_data(self.origin.valid_set()))