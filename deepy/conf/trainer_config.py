#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging as loggers
logging = loggers.getLogger(__name__)
from config import GeneralConfig

class TrainerConfig(GeneralConfig):
    """
    Training configuration container.
    """
    def __init__(self):
        super(TrainerConfig, self).__init__(logger=logging)
        object.__setattr__(self, "attrs", {
            # Training
            "validation_frequency": 1,
            "test_frequency": 10,
            "monitor_frequency": 1,
            "min_improvement": 0.,
            "patience": 20,

            # Optimization
            "method": "ADADELTA",

            # Regularization
            "update_l1": 0,
            "update_l2": 0,
            "weight_l1": 0,
            "weight_l2": 0,
            "hidden_l1": 0,
            "hidden_l2": 0,
            "contractive_l2": 0,
        })