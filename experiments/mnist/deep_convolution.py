#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging, os
logging.basicConfig(level=logging.INFO)

from deepy.dataset import MnistDataset, MiniBatches
from deepy.networks import NeuralClassifier
from deepy.layers import Convolution, Dense, Flatten, DimShuffle, Reshape, RevealDimension, Softmax, Dropout
from deepy.trainers import MomentumTrainer, LearningRateAnnealer

default_model = os.path.join(os.path.dirname(__file__), "models", "deep_conv.gz")

if __name__ == '__main__':
    model = NeuralClassifier(input_dim=28*28)
    model.stack(# Reshape to 3D tensor
                Reshape((-1, 28, 28)),
                # Add a new dimension for convolution
                DimShuffle((0, 'x', 1, 2)),
                Convolution((4, 1, 5, 5), activation="relu"),
                Dropout(0.15),
                Convolution((8, 4, 5, 5), activation="relu"),
                Dropout(0.1),
                Convolution((16, 8, 3, 3), activation="relu"),
                Flatten(),
                Dropout(0.1),
                # As dimension information was lost, reveal it to the pipe line
                RevealDimension(16),
                Dense(10, 'linear'),
                Softmax())

    trainer = MomentumTrainer(model)

    annealer = LearningRateAnnealer(trainer)

    mnist = MiniBatches(MnistDataset(), batch_size=20)

    trainer.run(mnist, controllers=[annealer])

    model.save_params(default_model)

