#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
An auto-encoder for compress MNIST images.
"""


import logging, os
logging.basicConfig(level=logging.INFO)

from deepy.dataset import MnistDataset, MiniBatches
from deepy.networks import AutoEncoder
from deepy.layers import Dense
from deepy.trainers import SGDTrainer, LearningRateAnnealer
from deepy.utils import shared_scalar

model_path = os.path.join(os.path.dirname(__file__), "models", "mnist_autoencoder.gz")

if __name__ == '__main__':
    model = AutoEncoder(input_dim=28 * 28, rep_dim=30)
    model.stack_encoders(Dense(50, 'tanh'), Dense(30))
    model.stack_decoders(Dense(50, 'tanh'), Dense(28 * 28))

    trainer = SGDTrainer(model, {'learning_rate': graph.shared(0.05), 'gradient_clipping': 3})

    mnist = MiniBatches(MnistDataset(for_autoencoder=True), batch_size=20)

    trainer.run(mnist, epoch_controllers=[LearningRateAnnealer()])

    model.save_params(model_path)