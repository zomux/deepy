#!/usr/bin/env python
# -*- coding: utf-8 -*-


import time
import logging

import numpy as np

from deepy.layers.recurrent import RecurrentLayer, RecurrentNetwork
from deepy.conf import NetworkConfig, TrainerConfig
from deepy.utils.functions import FLOATX
from deepy import SGDTrainer


logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    net_conf = NetworkConfig(input_size=6)
    net_conf.layers = [RecurrentLayer(size=10, activation='sigmoid', bptt=True)]

    trainer_conf = TrainerConfig()
    trainer_conf.learning_rate = 0.03
    trainer_conf.weight_l2 = 0.0001
    trainer_conf.hidden_l2 = 0.0001
    trainer_conf.monitor_frequency = trainer_conf.validation_frequency = trainer_conf.test_frequency = 1

    network = RecurrentNetwork(net_conf)
    trainer = SGDTrainer(network)

    data = np.array([[1,0,0,0,0,0],
                     [0,1,0,0,0,0],
                     [0,0,1,0,0,0],
                     [0,0,0,1,0,0],
                     [0,0,0,0,1,0],
                     [0,0,0,0,0,1],
                     [0,1,0,0,0,0],
                     [0,0,1,0,0,0],
                     [0,0,0,1,0,0],
                     [0,0,0,0,1,0],
                     [0,0,0,0,0,1],
                     [1,0,0,0,0,0]], dtype=FLOATX)

    targets = np.array([3,
                        3,
                        2,
                        2,
                         2,
                         2,
                         3,
                         3,
                         2,
                         2,
                         2,
                         3], dtype=FLOATX)

    test_data = np.array([[0,1,0,0,0,0],
                          [0,0,0,1,0,0],
                          [0,0,1,0,0,0],
                          [0,0,0,1,0,0],
                          [0,0,0,0,1,0],
                          [0,1,0,0,0,0],
                          [0,0,1,0,0,0]], dtype=FLOATX)

    test_targets = np.array([3,
                        3,
                        3,
                        3,
                         2,
                         3,
                         3,], dtype=FLOATX)

    train_set = valid_set = [(np.array(x[:3]), np.array(x[3:])) for x in zip(data[:-2], data[1:-1], data[2:], targets[:-2], targets[1:-1], targets[2:])]
    test_set = [(np.array(x[:3]), np.array(x[3:])) for x in zip(test_data[:-2], test_data[1:-1], test_data[2:], test_targets[:-2], test_targets[1:-1], test_targets[2:])]


    start_time = time.time()
    for k in list(trainer.train(train_set, valid_set, test_set=test_set)):
        pass
    print k
    trainer.test(0, test_set)
    end_time = time.time()


    print "elapsed time:", (end_time - start_time) / 60, "mins"