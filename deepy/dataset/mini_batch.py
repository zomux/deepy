#!/usr/bin/env python
# -*- coding: utf-8 -*-


from . import Dataset
import numpy as np

class MiniBatches(Dataset):

    def __init__(self, dataset, batch_size=20):
        self.origin = dataset
        self.size = batch_size
        self._train_set = None
        self._valid_set = None
        self._test_set = None

    def _yield_data(self, subset):
        for i in xrange(0, len(subset), self.size):
            yield map(np.array, list(zip(*subset[i:i + self.size])))

    def train_set(self):
        if not self.origin.train_set():
            return None
        if not self._train_set:
            self._train_set = list(self._yield_data(self.origin.train_set()))
        return self._train_set

    def test_set(self):
        if not self.origin.test_set():
            return None
        if not self._test_set:
            self._test_set = list(self._yield_data(self.origin.test_set()))
        return self._test_set

    def valid_set(self):
        if not self.origin.valid_set():
            return None
        if not self._valid_set:
            self._valid_set = list(self._yield_data(self.origin.valid_set()))
        return self._valid_set

    def train_size(self):
        return len(list(self.origin.train_set())) / self.size