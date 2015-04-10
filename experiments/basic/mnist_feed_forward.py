#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logging.basicConfig(level=logging.INFO)

from deepy.dataset import MnistDataset, MiniBatches
from deepy.networks import NeuralClassifier
from deepy.layers import Dense, Softmax
from deepy.trainers import MomentumTrainer, LearningRateAnnealer

if __name__ == '__main__':
    model = NeuralClassifier(input_dim=28*28)
    model.stack_layers(Dense(256, 'relu'),
                       Dense(256, 'relu'),
                       Dense(10, 'linear'),
                       Softmax())

    trainer = MomentumTrainer(model)

    annealer = LearningRateAnnealer(trainer)

    mnist = MiniBatches(MnistDataset(), batch_size=20)

    trainer.run(mnist.train_set(), mnist.valid_set(), test_set=mnist.test_set(), controllers=[annealer])

