#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from itertools import izip, izip_longest, imap


def pad_sequence(batch, pad_value=0, pad_side='right', output_mask=False):
    """
    Pad a sequence with a value either in the left or right side.
    """
    max_len = max(map(len, batch))
    mask = None
    if output_mask:
        mask = []
        for i in range(len(batch)):
            mask.append([1] * len(batch[i]) + [0] * (max_len - len(batch[i])))
        mask = np.array(mask, dtype="float32")
    if pad_side == 'right':
        new_batch = np.array(list(izip(*izip_longest(*batch, fillvalue=pad_value))))
    else:
        new_batch = np.array(list(izip(*izip_longest(*imap(lambda x: x[::-1], batch), fillvalue=pad_value))))[:, ::-1]
    if output_mask:
        return new_batch, mask
    else:
        return new_batch