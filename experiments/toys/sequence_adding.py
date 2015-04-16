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
import logging, os
import numpy as np
logging.basicConfig(level=logging.INFO)
from deepy.conf import TrainerConfig
from deepy.dataset import SequenceDataset, MiniBatches
from deepy.networks import NeuralRegressor
from deepy.layers import RNN, IRNN
from deepy.trainers import SGDTrainer, LearningRateAnnealer
from deepy.util import FLOATX

SEQUENCE_LEN = 100
rand = np.random.RandomState(3)

data = []
for _ in range(50000):
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

batch_set = MiniBatches(dataset, batch_size=32)

if __name__ == '__main__':

    model_file = os.path.join(os.path.dirname(__file__), "models", "sequence_adding_100_2.gz")

    model = NeuralRegressor(input_dim=2, input_tensor=3)
    model.stack(IRNN(hidden_size=100, output_size=1,
                    input_type="sequence", output_type="last_output",
                    output_activation="linear"))
    if os.path.exists(model_file):
        model.load_params(model_file)

    conf = TrainerConfig()
    conf.learning_rate = LearningRateAnnealer.learning_rate(0.01)
    conf.max_norm = 10
    conf.patience = 50
    trainer = SGDTrainer(model, conf)

    annealer = LearningRateAnnealer(trainer, patience=20)

    trainer.run(batch_set.train_set(), batch_set.valid_set(), controllers=[annealer])

    model.save_params(model_file)
    print "Identity matrix weight:"
    print model.first_layer().W_h.get_value().diagonal()
