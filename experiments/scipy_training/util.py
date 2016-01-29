#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
logging.basicConfig(level=logging.INFO)

from deepy.dataset import MnistDataset, MiniBatches
from deepy.networks import NeuralClassifier
from deepy.layers import Dense, Softmax
from deepy.trainers import LearningRateAnnealer, ScipyTrainer

def run(method, model_path):
    model = NeuralClassifier(input_dim=28 * 28)
    model.stack(Dense(128, 'relu'),
                Dense(128, 'relu'),
                Dense(10, 'linear'),
                Softmax())

    trainer = ScipyTrainer(model, method)

    annealer = LearningRateAnnealer(trainer)

    mnist = MiniBatches(MnistDataset(), batch_size=100)

    trainer.run(mnist, controllers=[annealer])

    model.save_params(model_path)