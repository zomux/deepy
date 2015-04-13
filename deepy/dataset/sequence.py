#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import BasicDataset
import numpy as np
from deepy.util import FLOATX

class SequenceDataset(BasicDataset):
    """
    Dataset class for sequences.
    """

    def __init__(self, train, valid=None, test=None):
        super(SequenceDataset, self).__init__(train, valid, test)

    def _pad_set(self, subset, side, length):
        row_size = subset[0][0][0].shape[0]
        new_set = []
        for x, y in subset:
            if len(x) > length:
                x = x[:length]
            elif len(x) < length:
                pad_length = length - len(x)
                pad_matrix = np.zeros((pad_length,row_size), dtype=FLOATX)
                if side == "left":
                    x = np.vstack([pad_matrix, x])
                elif side == "right":
                    x = np.vstack([x, pad_matrix])
                else:
                    return Exception("Side of padding must be 'left' or 'right'")
            new_set.append((x, y))
        return new_set

    def _pad(self, side, length):
        """
        Pad sequences to given length in the left or right side.
        """
        if self._train_set:
            self._train_set = self._pad_set(self._train_set, side, length)
        if self._valid_set:
            self._valid_set = self._pad_set(self._valid_set, side, length)
        if self._test_set:
            self._test_set = self._pad_set(self._test_set, side, length)

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