#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import BasicDataset
from padding import pad_dataset

class SequentialDataset(BasicDataset):
    """
    Dataset class for sequences.
    """

    def __init__(self, train, valid=None, test=None):
        super(SequentialDataset, self).__init__(train, valid, test)

    def _pad(self, side, length):
        """
        Pad sequences to given length in the left or right side.
        """
        if self._train_set:
            self._train_set = pad_dataset(self._train_set, side, length)
        if self._valid_set:
            self._valid_set = pad_dataset(self._valid_set, side, length)
        if self._test_set:
            self._test_set = pad_dataset(self._test_set, side, length)

    def pad_left(self, length):
        """
        Pad sequences to given length in the left side.
        """
        self._pad('left', length)

    def pad_right(self, length):
        """
        Pad sequences to given length in the left side.
        """
        self._pad('right', length)