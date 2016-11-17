#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from random import Random
from deepy.utils import onehot
from deepy.dataset import MiniBatches, BasicDataset
random = Random(3)

VECTOR_SIZE = 30
SEQUENCE_LENGTH = 5
DATA_SIZE = 1000

def random_vector():
    return onehot(VECTOR_SIZE, random.randint(0, VECTOR_SIZE - 1))

def get_data():
    data = []
    for _ in range(DATA_SIZE):
        sequence = []
        for _ in range(SEQUENCE_LENGTH):
            sequence.append(random_vector())
        data.append([np.vstack(sequence)])
    valid_size = int(DATA_SIZE * 0.1)
    return MiniBatches(BasicDataset(data[valid_size:], valid=data[:valid_size]))
