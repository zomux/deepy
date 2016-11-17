#!/usr/bin/env python
# -*- coding: utf-8 -*-


class DataProcessor(object):
    """
    An abstract class for data processor.
    """
    def process(self, split, epoch, dataset):
        return dataset