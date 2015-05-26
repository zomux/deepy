#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging as loggers
from config import GeneralConfig
from deepy.utils import FLOATX
import theano
import numpy as np
logging = loggers.getLogger(__name__)


class TrainerConfig(GeneralConfig):
    """
    Training configuration container.
    """
    def __init__(self):
        super(TrainerConfig, self).__init__(logger=logging)
        object.__setattr__(self, "attrs", {
            # Training
            "learning_rate": theano.shared(np.array(0.01, dtype=FLOATX)),
            "validation_frequency": 1,
            "test_frequency": 10,
            "monitor_frequency": 1,
            "min_improvement": 0.001,
            "max_iterations": 0,
            "patience": 20,

            # Optimization
            "method": "ADADELTA",
            "weight_bound": None,
            "avoid_nan": False,
            "gradient_tolerance": None,
            "gradient_clipping": None, # l1 or l2
            "max_norm": 10,

            # Regularization
            "update_l1": 0,
            "update_l2": 0,
            "weight_l1": 0,
            "weight_l2": 0.0001,
            "hidden_l1": 0,
            "hidden_l2": 0,
        })