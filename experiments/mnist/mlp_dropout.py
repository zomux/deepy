#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
For reference, this model should achieve 1.50% error rate, in 10 mins with i7 CPU (8 threads).
"""

import logging, os
logging.basicConfig(level=logging.INFO)

from deepy.dataset import MnistDataset, MiniBatches
from deepy.networks import NeuralClassifier
from deepy.layers import Dense, Softmax, Dropout
from deepy.trainers import MomentumTrainer, LearningRateAnnealer

default_model = os.path.join(os.path.dirname(__file__), "models", "mlp_dropout1.gz")

if __name__ == '__main__':
    model = NeuralClassifier(input_dim=28*28)
    model.stack(Dense(256, 'relu'),
                Dropout(0.5),
                Dense(256, 'relu'),
                Dropout(0.5),
                Dense(10, 'linear'),
                Softmax())

    trainer = MomentumTrainer(model)

    annealer = LearningRateAnnealer(trainer)

    mnist = MiniBatches(MnistDataset(), batch_size=20)

    trainer.run(mnist, controllers=[annealer])

    model.save_params(default_model)