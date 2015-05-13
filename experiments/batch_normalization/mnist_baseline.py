#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This experiment setting is described in http://arxiv.org/pdf/1502.03167v3.pdf.
MNIST MLP baseline model.
Gaussian initialization described in the paper did not convergence, I have no idea.
"""

import logging, os
logging.basicConfig(level=logging.INFO)

from deepy.dataset import MnistDataset, MiniBatches
from deepy.networks import NeuralClassifier
from deepy.layers import Dense, Softmax
from deepy.trainers import SGDTrainer

default_model = os.path.join(os.path.dirname(__file__), "models", "baseline1.gz")

if __name__ == '__main__':
    model = NeuralClassifier(input_dim=28*28)
    model.stack(Dense(100, 'sigmoid'),
                Dense(100, 'sigmoid'),
                Dense(100, 'sigmoid'),
                Dense(10, 'linear'),
                Softmax())

    trainer = SGDTrainer(model)

    batches = MiniBatches(MnistDataset(), batch_size=60)

    trainer.run(batches, controllers=[])

    model.save_params(default_model)