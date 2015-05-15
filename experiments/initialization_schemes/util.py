#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logging.basicConfig(level=logging.INFO)

from deepy.dataset import MnistDataset, MiniBatches
from deepy.networks import NeuralClassifier
from deepy.layers import Dense, Softmax
from deepy.trainers import MomentumTrainer, LearningRateAnnealer

"""
Build a neural network with 6 dense layers with ReLU.
"""

def run(initializer, model_path):
    model = NeuralClassifier(input_dim=28*28)
    for _ in range(6):
        model.stack(Dense(128, 'relu', init=initializer))
    model.stack(Dense(10, 'linear'),
                Softmax())

    trainer = MomentumTrainer(model)

    annealer = LearningRateAnnealer(trainer)

    mnist = MiniBatches(MnistDataset(), batch_size=20)

    trainer.run(mnist, controllers=[annealer])

    model.save_params(model_path)