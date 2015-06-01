#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging, os
logging.basicConfig(level=logging.INFO)

from deepy.dataset import MnistDataset, MiniBatches
from deepy.networks import NeuralClassifier
from deepy.layers import Dense, Softmax
from deepy.trainers import MomentumTrainer, LearningRateAnnealer

default_model = os.path.join(os.path.dirname(__file__), "models", "mlp1.gz")

if __name__ == '__main__':
    model = NeuralClassifier(input_dim=28*28)
    model.stack(Dense(256, 'relu'),
                Dense(256, 'relu'),
                Dense(10, 'linear'),
                Softmax())

    trainer = MomentumTrainer(model, {"weight_l2": 0.001})

    annealer = LearningRateAnnealer(trainer)

    mnist = MiniBatches(MnistDataset(), batch_size=20)

    trainer.run(mnist, controllers=[annealer])

    model.save_params(default_model)