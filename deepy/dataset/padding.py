#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from deepy.utils import FLOATX

def pad_dataset(subset, side, length):
    """
    Pad data set to specified length.
    Parameters:
        length - max length, a just to the max length in the batch if length is -1
    """
    assert length == -1 or length > 0
    if type(subset[0][0][0]) in [float, int]:
        return _pad_2d(subset, side, length)
    else:
        return _pad_3d(subset, side, length)

def _pad_2d(subset, side, length):
    new_set = []
    max_len = max([len(x) for x, _ in subset]) if length == -1 else length
    for x, y in subset:
        if len(y) > max_len:
            y = y[:max_len]
        elif len(y) < max_len:
            if side == "left":
                y = [0 for _ in range(max_len - len(y))] + y
            elif side == "right":
                y = y + [0 for _ in range(max_len - len(y))]
        if len(x) > max_len:
            x = x[:max_len]
        elif len(x) < max_len:
            if side == "left":
                x = [0 for _ in range(max_len - len(x))] + x
            elif side == "right":
                x = x + [0 for _ in range(max_len - len(x))]
        new_set.append((x, y))
    return new_set

def _pad_3d(subset, side, length):
    row_size = subset[0][0][0].shape[0]
    new_set = []
    max_len = max([len(x) for x, _ in subset]) if length == -1 else length
    for x, y in subset:
        if type(y) == list:
            # Clip target vector
            if len(y) > max_len:
                y = y[:max_len]
            elif len(y) < max_len:
                if side == "left":
                    y = [0 for _ in range(max_len - len(y))] + y
                elif side == "right":
                    y = y + [0 for _ in range(max_len - len(y))]
        if len(x) > max_len:
            x = x[:max_len]
        elif len(x) < max_len:
            pad_length = max_len - len(x)
            pad_matrix = np.zeros((pad_length,row_size), dtype=FLOATX)
            if side == "left":
                x = np.vstack([pad_matrix, x])
            elif side == "right":
                x = np.vstack([x, pad_matrix])
            else:
                return Exception("Side of padding must be 'left' or 'right'")
        new_set.append((x, y))
    return new_set