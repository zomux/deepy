#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging, os, random
import numpy as np
logging.basicConfig(level=logging.INFO)

from deepy.dataset import SequentialDataset, MiniBatches
from deepy.networks import NeuralClassifier
from deepy.layers import RNN, Dense, Softmax
from deepy.trainers import SGDTrainer, LearningRateAnnealer
from deepy.utils import FLOATX

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
    word_matrix = np.vstack(char_vectors)
    data.append((word_matrix, label))

# Shuffle the data
random.Random(3).shuffle(data)

# Separate data
valid_size = int(len(data) * 0.15)
train_set = data[valid_size:]
valid_set = data[:valid_size]

dataset = SequentialDataset(train_set, valid=valid_set)
dataset.pad_left(20)
dataset.report()

batch_set = MiniBatches(dataset)

if __name__ == '__main__':
    model = NeuralClassifier(input_dim=26, input_tensor=3)
    model.stack(RNN(hidden_size=30, input_type="sequence", output_type="sequence", vector_core=0.1),
                       RNN(hidden_size=30, input_type="sequence", output_type="sequence", vector_core=0.3),
                       RNN(hidden_size=30, input_type="sequence", output_type="sequence", vector_core=0.6),
                       RNN(hidden_size=30, input_type="sequence", output_type="one", vector_core=0.9),
                       Dense(4),
                       Softmax())

    trainer = SGDTrainer(model)

    annealer = LearningRateAnnealer(trainer)

    trainer.run(batch_set.train_set(), batch_set.valid_set(), controllers=[annealer])

