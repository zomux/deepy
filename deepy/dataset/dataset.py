#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import collections

class Dataset(object):
    """
    Abstract dataset class.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def train_set(self):
        """
        :rtype: list of tuple
        """

    def valid_set(self):
        """
        :rtype: list of tuple
        """

    def test_set(self):
        """
        :rtype: list of tuple
        """

    def train_size(self):
        """
        Return size of training data. (optional)
        :rtype: number
        """
        train_set = self.train_set()
        if isinstance(train_set, collections.Iterable):
            return len(list(train_set))
        else:
            return None