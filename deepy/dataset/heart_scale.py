#!/usr/bin/env python
# -*- coding: utf-8 -*-


from deepy.dataset import AbstractDataset
from nlpy.util import FeatureContainer, internal_resource
import numpy as np

class HeartScaleDataset(AbstractDataset):

    def __init__(self, target_format=None):
        super(HeartScaleDataset, self).__init__(target_format)
        feature = FeatureContainer(internal_resource("dataset/heart_scale.txt"))
        self.data = feature.data
        self.targets = feature.targets
        self._target_size = 2

    def train_set(self):
        return [(self.data[:150], np.array(map(self._target_map, self.targets[:150])))]

    def valid_set(self):
        return [(self.data[150:200], np.array(map(self._target_map, self.targets[150:200])))]

    def test_set(self):
        return [(self.data[200:], np.array(map(self._target_map, self.targets[200:])))]