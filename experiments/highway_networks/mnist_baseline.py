#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This experiment setting is described in following paper:
http://arxiv.org/abs/1505.00387 .

Classify MNIST digits using a very deep think network.
Plain deep networks are very hard to be trained, as shown in this case.

But we should notice that if highway layers just learn to pass information forward,
in other words, just be transparent layers, then they would be meaningless.
"""

import logging, os
logging.basicConfig(level=logging.INFO)

from deepy.dataset import MnistDataset, MiniBatches
from deepy.networks import NeuralClassifier
from deepy.layers import Dense, Softmax
from deepy.trainers import MomentumTrainer, LearningRateAnnealer

model_path = os.path.join(os.path.dirname(__file__), "models", "baseline1.gz")

if __name__ == '__main__':
    model = NeuralClassifier(input_dim=28*28)
    for _ in range(20):
        model.stack(Dense(71, 'relu'))
    model.stack(Dense(10, 'linear'),
                Softmax())

    trainer = MomentumTrainer(model)

    mnist = MiniBatches(MnistDataset(), batch_size=20)

    trainer.run(mnist, controllers=[LearningRateAnnealer(trainer)])

    model.save_params(model_path)