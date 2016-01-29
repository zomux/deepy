#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This experiment setting is described in http://arxiv.org/pdf/1502.03167v3.pdf.
MNIST MLP model with batch normalization.
In my experiment, it turns out the improvement of valid data stopped after 37 epochs. (See models/batch_norm1.log)
"""

import logging, os
logging.basicConfig(level=logging.INFO)

from deepy.dataset import MnistDataset, MiniBatches
from deepy.networks import NeuralClassifier
from deepy.layers import Dense, Softmax, BatchNormalization
from deepy.trainers import SGDTrainer

default_model = os.path.join(os.path.dirname(__file__), "models", "batch_norm1.gz")

if __name__ == '__main__':
    model = NeuralClassifier(input_dim=28 * 28)
    model.stack(Dense(100, 'sigmoid'),
                BatchNormalization(),
                Dense(100, 'sigmoid'),
                BatchNormalization(),
                Dense(100, 'sigmoid'),
                BatchNormalization(),
                Dense(10, 'linear'),
                Softmax())

    trainer = SGDTrainer(model)

    batches = MiniBatches(MnistDataset(), batch_size=60)

    trainer.run(batches, controllers=[])

    model.save_params(default_model)