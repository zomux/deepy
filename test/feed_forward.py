#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest

import numpy as np
from deepy import NeuralRegressor
from deepy.trainers.trainer import SGDTrainer


class FeedForwardTest(unittest.TestCase):

    def test(self):
        from deepy.dataset import HeartScaleDataset
        from deepy.conf import NetworkConfig
        from deepy import NeuralLayer
        import logging
        logging.basicConfig(level=logging.INFO)
        conf = NetworkConfig(input_size=13)
        conf.layers = [NeuralLayer(10), NeuralLayer(5), NeuralLayer(1, 'linear')]
        ff = NeuralRegressor(conf)
        t = SGDTrainer(ff)
        train_set = [(np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13]]), np.array([[1,0]]))]
        a = [HeartScaleDataset(single_target=True).train_set()]
        b = [HeartScaleDataset(single_target=True).valid_set()]
        for k in list(t.train(a, b)):
            pass
        print k


if __name__ == '__main__':
    unittest.main()