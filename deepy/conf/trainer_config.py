#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging as loggers
from config import GeneralConfig
from deepy.utils import FLOATX
import theano
import numpy as np
logging = loggers.getLogger(__name__)

DEFAULT_TRAINER_SETTING = {
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
    "gradient_clipping": None, # L2 clipping value

    # Regularization
    "update_l1": 0,
    "update_l2": 0,
    "weight_l1": 0,
    "weight_l2": 0,
    "hidden_l1": 0,
    "hidden_l2": 0,
}

class TrainerConfig(GeneralConfig):
    """
    Training configuration container.
    """
    def __init__(self, settingMap=None):
        super(TrainerConfig, self).__init__(logger=logging)

        settings = DEFAULT_TRAINER_SETTING
        if isinstance(settingMap, dict):
            settings.update(settingMap)

        for key, value in settings.items():
            self.attrs[key] = value