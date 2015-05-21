#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging as loggers
logging = loggers.getLogger(__name__)
import os
import tempfile
import numpy as np
import urllib
from basic import BasicDataset

URL_MAP = {
    "train": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat",
    "valid": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat",
    "test": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat"
}

PATH_MAP = {
    "train": os.path.join(tempfile.gettempdir(), "binarized_mnist_train.npy"),
    "valid": os.path.join(tempfile.gettempdir(), "binarized_mnist_valid.npy"),
    "test": os.path.join(tempfile.gettempdir(), "binarized_mnist_test.npy")
}

class BinarizedMnistDataset(BasicDataset):

    def __init__(self):
        for name, url in URL_MAP.items():
            local_path = PATH_MAP[name]
            if not os.path.exists(local_path):
                logging.info("downloading %s dataset of binarized MNIST")
                np.save(local_path, np.loadtxt(urllib.urlretrieve(url)[0]))
        train_set = [(x,) for x in np.load(PATH_MAP['train'])]
        valid_set = [(x,) for x in np.load(PATH_MAP['valid'])]
        test_set = [(x,) for x in np.load(PATH_MAP['test'])]
        super(BinarizedMnistDataset, self).__init__(train_set, valid=valid_set, test=test_set)
