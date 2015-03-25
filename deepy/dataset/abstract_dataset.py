#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

class AbstractDataset(object):

    def __init__(self, target_format=None):
        self.target_format = target_format
        self._target_size = 0

    def _target_map(self, i):
        if self.target_format == 'vector' and self._target_size > 0:
            l = [0.] * self._target_size
            l[i] = 1.
            return l
        if self.target_format == 'tuple':
            return [i]
        elif self.target_format == 'number':
            return i
        else:
            return i

    def train_set(self):
        """
        :rtype: tuple
        """

    def valid_set(self):
        """
        :rtype: tuple
        """

    def test_set(self):
        """
        :rtype: tuple
        """