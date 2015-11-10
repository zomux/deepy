#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
For reference, this model should achieve 1.50% error rate, in 10 mins with i7 CPU (8 threads).
"""

import logging, os
logging.basicConfig(level=logging.INFO)

import numpy as np

from deepy.dataset import MnistDataset, MiniBatches
from deepy.networks import NeuralClassifier
from deepy.layers import Softmax, Dropout, Maxout, Dense
from deepy.trainers import MomentumTrainer, ExponentialLearningRateAnnealer
from deepy.utils import UniformInitializer, shared_scalar

default_model = os.path.join(os.path.dirname(__file__), "models", "mlp_maxout1.gz")

L2NORM_LIMIT = 1.9365
EPSILON = 1e-7

def clip_param_norm():
    for param in model.parameters:
        if param.name.startswith("W"):
            l2_norms = np.sqrt(np.sum(param.get_value() ** 2, axis=0, keepdims=True))
            desired_norms = np.clip(l2_norms, 0, L2NORM_LIMIT)
            scale = (desired_norms + EPSILON) / (l2_norms + EPSILON)
            param.set_value(param.get_value() * scale)

if __name__ == '__main__':
    model = NeuralClassifier(input_dim=28*28)
    model.training_callbacks.append(clip_param_norm)
    model.stack(Dropout(0.2),
                Maxout(240, num_pieces=5, init=UniformInitializer(.005)),
                Maxout(240, num_pieces=5, init=UniformInitializer(.005)),
                Dense(10, 'linear', init=UniformInitializer(.005)),
                Softmax())


    trainer = MomentumTrainer(model, {"learning_rate": shared_scalar(0.01),
                                      "momentum": 0.5})

    annealer = ExponentialLearningRateAnnealer(trainer, debug=True)

    mnist = MiniBatches(MnistDataset(), batch_size=100)

    trainer.run(mnist, controllers=[annealer])

    model.save_params(default_model)