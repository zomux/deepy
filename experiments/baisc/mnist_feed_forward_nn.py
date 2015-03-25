#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
import time

from deepy.dataset import MnistDataset, MiniBatches
from deepy.conf import NetworkConfig, TrainerConfig
from deepy import NeuralLayer, NeuralClassifier, SGDTrainer


logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    net_conf = NetworkConfig(input_size=28*28)
    net_conf.layers = [NeuralLayer(500, 'sigmoid'), NeuralLayer(10, 'softmax')]

    trainer_conf = TrainerConfig()
    trainer_conf.learning_rate = 0.01
    trainer_conf.weight_l2 = 0.0001
    trainer_conf.hidden_l2 = 0.0001

    network = NeuralClassifier(net_conf)
    trainer = SGDTrainer(network)

    mnist = MiniBatches(MnistDataset(target_format='number'), batch_size=20)

    start_time = time.time()
    for k in list(trainer.train(mnist.train_set(), mnist.valid_set(), test_set=mnist.test_set())):
        pass
    print k
    trainer.test(0, mnist.test_set())
    end_time = time.time()
    print "elapsed time:", (end_time - start_time) / 60, "mins"