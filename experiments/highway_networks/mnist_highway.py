#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This experiment setting is described in following paper:
http://arxiv.org/abs/1505.00387 .

With highway network layers, Very deep networks (20 layers here) can be trained properly.
"""

import logging, os
logging.basicConfig(level=logging.INFO)

from deepy.dataset import MnistDataset, MiniBatches
from deepy.networks import NeuralClassifier
from deepy.layers import Dense, Softmax
from deepy.trainers import MomentumTrainer, LearningRateAnnealer
from highway_layer import HighwayLayer

model_path = os.path.join(os.path.dirname(__file__), "models", "highway1.gz")

if __name__ == '__main__':
    model = NeuralClassifier(input_dim=28 * 28)
    model.stack(Dense(50, 'relu'))
    for _ in range(20):
        model.stack(HighwayLayer(activation='relu'))
    model.stack(Dense(10, 'linear'),
                Softmax())

    trainer = MomentumTrainer(model)

    mnist = MiniBatches(MnistDataset(), batch_size=20)

    trainer.run(mnist, controllers=[LearningRateAnnealer(trainer)])

    model.save_params(model_path)