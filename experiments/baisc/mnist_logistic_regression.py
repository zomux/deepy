#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from nlpy.classification import LogisticRegression

if __name__ == '__main__':
    from deepy.dataset import MnistDataset
    import logging
    logging.basicConfig(level=logging.DEBUG)
    lr = LogisticRegression(batch_size=10)
    lr.train(MnistDataset())