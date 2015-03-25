#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
import tempfile
import os
import gzip
import urllib
import cPickle

from deepy.dataset import AbstractDataset
import numpy as np


logging = logging.getLogger(__name__)

MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl.gz"

class MnistDataset(AbstractDataset):

    def __init__(self, target_format=None):
        super(MnistDataset, self).__init__(target_format)
        self._target_size = 10
        logging.info("loading minst data")
        path = os.path.join(tempfile.gettempdir(), "mnist.pkl.gz")
        if not os.path.exists(path):
            logging.info("downloading minst data")
            urllib.urlretrieve (MNIST_URL, path)
        self._train_set, self._valid_set, self._test_set = cPickle.load(gzip.open(path, 'rb'))
        logging.info("[mnist] training data size: %d" % len(self._train_set[0]))
        logging.info("[mnist] valid data size: %d" % len(self._valid_set[0]))
        logging.info("[mnist] test data size: %d" % len(self._test_set[0]))

    def train_set(self):
        data, target = self._train_set
        return [(data,  np.array(map(self._target_map, target)))]

    def valid_set(self):
        data, target = self._valid_set
        return [(data,  np.array(map(self._target_map, target)))]

    def test_set(self):
        data, target = self._test_set
        return [(data,  np.array(map(self._target_map, target)))]