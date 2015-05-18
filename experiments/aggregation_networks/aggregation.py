#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This experiment setting is described in following paper:
http://arxiv.org/abs/1505.00387 .

With highway network layers, Very deep networks (20 layers here) can be trained properly.
"""

import logging, os
logging.basicConfig(level=logging.INFO)

from deepy import *
from aggregation_layer import AggregationLayer

model_path = os.path.join(os.path.dirname(__file__), "models", "aggregation1.gz")

if __name__ == '__main__':
    model = NeuralClassifier(input_dim=28*28)
    model.stack(AggregationLayer(50, layers=20, init=KaimingHeInitializer()),
                Dense(10, 'linear', init=KaimingHeInitializer()),
                Softmax())

    trainer = SGDTrainer(model)

    mnist = MiniBatches(MnistDataset(), batch_size=20)

    trainer.run(mnist, controllers=[LearningRateAnnealer(trainer)])

    model.save_params(model_path)