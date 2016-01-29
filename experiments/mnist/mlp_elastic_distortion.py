#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This code doubles the train data size by
appending a transformed image for each training data.

WARNING: Elastic distortion function is slow, if you plan to do some experiments on it,
better use multi-processing.
"""

import logging, os
logging.basicConfig(level=logging.INFO)

from deepy.dataset import MnistDataset, MiniBatches, BasicDataset
from deepy.networks import NeuralClassifier
from deepy.layers import Dense, Softmax
from deepy.trainers import MomentumTrainer, LearningRateAnnealer
from deepy.utils import elastic_distortion, global_rand

default_model = os.path.join(os.path.dirname(__file__), "models", "mlp_distortion1.gz")

mnist = MnistDataset()

logging.info("transforming images with elastic distortion")

expanded_train_set = []

for img, label in mnist.train_set():
    expanded_train_set.append((img, label))
    original_img = (img * 256).reshape((28, 28))
    transformed_img = (elastic_distortion(original_img) / 256).flatten()
    expanded_train_set.append((transformed_img, label))

global_rand.shuffle(expanded_train_set)

expanded_mnist = BasicDataset(train=expanded_train_set, valid=mnist.valid_set(), test=mnist.test_set())

logging.info("expanded training data size: %d" % len(expanded_train_set))

if __name__ == '__main__':
    model = NeuralClassifier(input_dim=28 * 28)
    model.stack(Dense(256, 'relu'),
                Dense(256, 'relu'),
                Dense(10, 'linear'),
                Softmax())

    trainer = MomentumTrainer(model)

    annealer = LearningRateAnnealer(trainer)

    mnist = MiniBatches(expanded_mnist, batch_size=20)

    trainer.run(mnist, controllers=[annealer])

    model.save_params(default_model)