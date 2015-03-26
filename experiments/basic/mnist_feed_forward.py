#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
logging.basicConfig(level=logging.INFO)

from deepy.dataset import MnistDataset, MiniBatches
from deepy import NetworkConfig, TrainerConfig
from deepy import NeuralLayer, NeuralClassifier
from deepy.trainers import MomentumTrainer, LearningRateAnnealer
from deepy.util import Timer

if __name__ == '__main__':
    net_conf = NetworkConfig(input_size=28*28)
    net_conf.layers = [NeuralLayer(256, 'tanh'), NeuralLayer(256, 'tanh'), NeuralLayer(10, 'softmax')]

    trainer_conf = TrainerConfig()
    trainer_conf.learning_rate = LearningRateAnnealer.learning_rate(0.01)
    trainer_conf.weight_l2 = 0.0001
    trainer_conf.hidden_l2 = 0.0001

    network = NeuralClassifier(net_conf)
    trainer = MomentumTrainer(network, config=trainer_conf)
    trainer_conf.report()

    annealer = LearningRateAnnealer(trainer)

    mnist = MiniBatches(MnistDataset(), batch_size=20)

    timer = Timer()
    for _ in trainer.train(mnist.train_set(), mnist.valid_set(), test_set=mnist.test_set()):
        if annealer.invoke():
            break

    timer.report()
