#!/usr/bin/env python
# -*- coding: utf-8 -*-


from . import MiniBatches
from padding import pad_dataset
import numpy as np

PADDING_SIDE = "right"

class SequentialMiniBatches(MiniBatches):
    """
    Mini batch class for sequential data.
    """

    def __init__(self, dataset, batch_size=20, padding_length=-1):
        super(SequentialMiniBatches, self).__init__(dataset, batch_size=batch_size)
        self.padding_length = padding_length

    def _yield_data(self, subset):
        for i in xrange(0, len(subset), self.size):
            x_set, y_set = [], []
            batch = pad_dataset(subset[i:i + self.size], PADDING_SIDE, self.padding_length)
            for x, y in batch:
                x_set.append(x)
                y_set.append(y)
            x_set = np.array(x_set)
            y_set = np.array(y_set)
            yield x_set, y_set