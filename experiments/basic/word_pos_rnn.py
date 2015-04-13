#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging, os, random
import numpy as np
logging.basicConfig(level=logging.INFO)

from deepy import NetworkConfig
from deepy.networks import NeuralClassifier
from deepy.layers import RNN, Dense, Softmax
from deepy.trainers import MomentumTrainer, LearningRateAnnealer
from deepy.util import FLOATX

WORD_POS_RESOURCE = os.path.abspath(os.path.dirname(__file__)) + os.sep + "resources/word_pos.txt"
CLASSES = ["N", "V", "ADJ", "ADV"]

# Create training data
# Raw data format: word pos_label
# Convert it to padded matrix, class pair
data = []
for line in open(WORD_POS_RESOURCE).readlines():
    word, pos = line.strip().split(" ")
    label = CLASSES.index(pos)
    char_vectors = []
    for char in word.lower():
        char_code = ord(char)
        if char_code >= ord("a") or char_code <= ord("z"):
            # Create one-hot vector
            char_vectors.append(np.eye(1, 26, char_code - ord("a"), dtype=FLOATX)[0])
        if len(char_vectors) >= 20:
            continue
    # Left-pad vectors
    while len(char_vectors) < 20:
        char_vectors.insert(0, np.zeros(26, dtype=FLOATX))
    word_matrix = np.vstack(char_vectors)
    data.append((word_matrix, label))

# Shuffle the data
random.Random(3).shuffle(data)

# Make mini-batches
batches = []
batch_size = 5
for i in range(0, len(data), batch_size):
    batch_x, batch_y = [], []
    for x, y in data[i: i+batch_size]:
        batch_x.append(x.reshape(1, 20, 26))
        batch_y.append(y)

    batches.append((np.vstack(batch_x), np.hstack(batch_y)))

# Separate data

valid_batches_size = int(len(batches) * 0.15)
train_batches = batches[valid_batches_size:]
valid_batches = batches[:valid_batches_size]
logging.info("Training data batches: %d, Valid data batches: %d" % (len(train_batches), len(valid_batches)))


if __name__ == '__main__':
    network_config = NetworkConfig()
    network_config.input_tensor = 3
    model = NeuralClassifier(input_dim=26, config=network_config)
    model.stack_layers(RNN(hidden_size=50, input_type="sequence", output_type="last_hidden"),
                       Dense(4),
                       Softmax())

    trainer = MomentumTrainer(model)

    annealer = LearningRateAnnealer(trainer)

    trainer.run(train_batches, valid_batches, controllers=[annealer])

