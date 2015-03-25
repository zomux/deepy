#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from deepy.dataset.abstract_dataset import AbstractDataset

class MiniBatches(AbstractDataset):

    def __init__(self, dataset, batch_size=20):
        self.origin = dataset
        self.size = batch_size

    def _yield_data(self, data, targets):
        for i in xrange(0, len(data), self.size):
            yield data[i:i + self.size], targets[i:i + self.size]

    def train_set(self):
        data, targets = self.origin.train_set()[0]
        return list(self._yield_data(data, targets))

    def test_set(self):
        data, targets = self.origin.test_set()[0]
        return list(self._yield_data(data, targets))

    def valid_set(self):
        data, targets = self.origin.valid_set()[0]
        return list(self._yield_data(data, targets))