#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from itertools import izip, izip_longest

def pad_sequence(batch, pad_value=0, output_mask=True, length=None):
    if length:
        max_len = length
    else:
        max_len = max(map(len, batch))
    mask = None
    if output_mask:
        mask = []
        for i in range(len(batch)):
            mask.append([1] * len(batch[i]) + [0] * (max_len - len(batch[i])))
        mask = np.array(mask, dtype="float32")
    if length:
        new_batch = []
        for i in range(len(batch)):
            new_row = list(batch[i]) + [pad_value] * (max_len - len(batch[i]))
            new_batch.append(new_row)
        new_batch = np.array(new_batch)
    else:
        new_batch = np.array(list(izip(*izip_longest(*batch, fillvalue=pad_value))))
    return new_batch, mask