#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import Dataset
from deepy.utils import FLOATX
import numpy as np

import logging as loggers
logging = loggers.getLogger(__name__)

class BasicDataset(Dataset):
    """
    Basic dataset class.
    """

    def __init__(self, train, valid=None, test=None):
        self._train_set = train
        self._valid_set = valid
        self._test_set = test

    def train_set(self):
        return self._train_set

    def valid_set(self):
        return self._valid_set

    def test_set(self):
        return self._test_set

    def map(self, func):
        """
        Process all data with given function.
        The scheme of function should be x,y -> x,y.
        """
        if self._train_set:
            self._train_set = map(func, self._train_set)
        if self._valid_set:
            self._valid_set = map(func, self._valid_set)
        if self._test_set:
            self._test_set = map(func, self._test_set)

    def _vectorize_set(self, subset, size):
        newset = []
        for x, y in subset:
            l = [0.] * size
            l[y] = 1.
            new_y = np.array(l, dtype=FLOATX)
            newset.append((x, new_y))
        return newset

    def vectorize_target(self, size):
        """
        Make targets be one-hot vectors.
        """
        if self._train_set:
            self._train_set = self._vectorize_set(self._train_set, size)
        if self._valid_set:
            self._valid_set = self._vectorize_set(self._valid_set, size)
        if self._test_set:
            self._test_set = self._vectorize_set(self._test_set, size)

    def report(self):
        """
        Print dataset statistics.
        """
        logging.info("%s train=%d valid=%d test=%d" % (self.__class__.__name__,
                                                       len(self._train_set) if self._train_set else 0,
                                                       len(self._valid_set) if self._valid_set else 0,
                                                       len(self._test_set) if self._test_set else 0))


