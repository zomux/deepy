#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sequence adding problem.
---

This problem is described in http://arxiv.org/abs/1504.00941 .
Each item of the sequence contains two units,
the first one is real value, and the second-one 1 or 0.

Train the recurrent network to return the sum of all first-unit values
with 1 in the second unit.
"""
import logging, os, random
import numpy as np
logging.basicConfig(level=logging.INFO)

from deepy.dataset import SequenceDataset, MiniBatches
from deepy.networks import NeuralRegressor
from deepy.layers import RNN, Dense
from deepy.trainers import MomentumTrainer, LearningRateAnnealer
from deepy.util import FLOATX, IdentityInitializer, GaussianInitializer

SEQUENCE_LEN = 30
rand = np.random.RandomState(3)

data = []
for _ in range(10000):
    sequence = []
    sum = 0.0
    selected_items = rand.choice(range(SEQUENCE_LEN), 2)
    for i in range(SEQUENCE_LEN):
        a = rand.uniform(0, 1)
        b = 1 if i in selected_items else 0
        if b == 1:
            sum += a
        sequence.append(np.array([a, b], dtype=FLOATX))
    sequence = np.vstack(sequence)
    sum = np.array([sum], dtype=FLOATX)
    data.append((sequence, sum))

# Separate data
valid_size = int(1000)
train_set = data[valid_size:]
valid_set = data[:valid_size]

dataset = SequenceDataset(train_set, valid=valid_set)
dataset.report()

batch_set = MiniBatches(dataset, batch_size=16)

if __name__ == '__main__':
    model = NeuralRegressor(input_dim=2, input_tensor=3)
    model.stack_layers(RNN(hidden_size=100, input_type="sequence", output_type="last_hidden",
                           hidden_initializer=IdentityInitializer(),
                           initializer=GaussianInitializer(deviation=0.001)),
                       Dense(1))

    trainer = MomentumTrainer(model)

    annealer = LearningRateAnnealer(trainer)

    trainer.run(batch_set.train_set(), batch_set.valid_set(), controllers=[annealer])
