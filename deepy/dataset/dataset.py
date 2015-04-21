#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Dataset(object):
    """
    Abstract dataset class.
    """

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