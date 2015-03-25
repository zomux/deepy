#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

class TrainerConfig(object):

    def __init__(self):
        self.validation_frequency = 10
        self.test_frequency = 50
        self.monitor_frequency = 50

        self.min_improvement = 0.
        self.patience = 20

        self.momentum = 0.9
        self.learning_rate = 1e-4

        # Regularization
        self.update_l1 = 0
        self.update_l2 = 0
        self.weight_l1 = 0
        self.weight_l2 = 0
        self.hidden_l1 = 0
        self.hidden_l2 = 0
        self.contractive_l2 = 0
