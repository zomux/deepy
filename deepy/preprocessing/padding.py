#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def pad_sequence(batch, pad_value=0, pad_side='right', output_mask=False, length=None, dtype="int32"):
    """
    Pad a sequence with a value either in the left or right side.
    """
    import deepy as D
    if length is not None:
        max_len = length
    else:
        max_len = max(map(len, batch))
    if output_mask:
        mask = np.zeros((len(batch), max_len), dtype=D.FLOATX)
    else:
        mask = None
    new_batch = np.zeros((len(batch), max_len), dtype=dtype)
    if pad_value != 0:
        new_batch.fill(pad_value)
    for i in range(len(batch)):
        if pad_side == 'right':
            new_batch[i, :len(batch[i])] = batch[i]
        else:
            new_batch[i, -len(batch[i]):] = batch[i]
        if output_mask:
            if pad_side == 'right':
                mask[i, :len(batch[i])].fill(1.)
            else:
                mask[i, -len(batch[i]):].fill(1.)
    if output_mask:
        return new_batch, mask
    else:
        return new_batch
