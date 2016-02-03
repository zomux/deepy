#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging, os
logging.basicConfig(level=logging.INFO)

# MNIST Multi-layer model with dropout.
from deepy.dataset import MnistDataset, MiniBatches
from deepy.networks import NeuralClassifier
from deepy.layers import Dense, Softmax, Dropout
from deepy.trainers import MomentumTrainer, LearningRateAnnealer, AdaDeltaTrainer
from deepy.utils import shared_scalar

model_path = os.path.join(os.path.dirname(__file__), "models", "tutorial1.gz")

if __name__ == '__main__':
    model = NeuralClassifier(input_dim=28 * 28)
    model.stack(Dense(256, 'relu'),
                Dropout(0.2),
                Dense(256, 'relu'),
                Dropout(0.2),
                Dense(10, 'linear'),
                Softmax())

    mnist = MiniBatches(MnistDataset(), batch_size=20)

    trainer = MomentumTrainer(model, {"learning_rate": shared_scalar(0.01)})

    annealer = LearningRateAnnealer(trainer)

    trainer.run(mnist, controllers=[annealer])

    model.save_params(model_path)


