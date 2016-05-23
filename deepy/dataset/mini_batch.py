#!/usr/bin/env python
# -*- coding: utf-8 -*-


from . import Dataset
import numpy as np

class MiniBatches(Dataset):
    """
    Convert data into mini-batches.
    """

    def __init__(self, dataset, batch_size=20, cache=True):
        self.origin = dataset
        self.size = batch_size
        self._cached_train_set = None
        self._cached_valid_set = None
        self._cached_test_set = None
        self.cache = cache

    def _yield_data(self, subset):
        for i in xrange(0, len(subset), self.size):
            yield map(np.array, list(zip(*subset[i:i + self.size])))

    def train_set(self):
        if self.cache and self._cached_train_set is not None:
            return self._cached_train_set

        data_generator = self._yield_data(self.origin.train_set())
        if data_generator is None:
            return None
        if self.cache:
            self._cached_train_set = list(data_generator)
            return self._cached_train_set
        else:
            return data_generator

    def test_set(self):
        if self.cache and self._cached_test_set is not None:
            return self._cached_test_set

        data_generator = self._yield_data(self.origin.test_set())
        if data_generator is None:
            return None
        if self.cache:
            self._cached_test_set = list(data_generator)
            return self._cached_test_set
        else:
            return data_generator

    def valid_set(self):
        if self.cache and self._cached_valid_set is not None:
            return self._cached_valid_set

        data_generator = self._yield_data(self.origin.valid_set())
        if data_generator is None:
            return None
        if self.cache:
            self._cached_valid_set = list(data_generator)
            return self._cached_valid_set
        else:
            return data_generator

    def train_size(self):
        train_size = self.origin.train_size()
        if train_size is None:
            train_size = len(list(self.origin.train_set()))
        return train_size / self.size