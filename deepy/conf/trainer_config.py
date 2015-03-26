#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging as loggers

logging = loggers.getLogger(__name__)

class TrainerConfig(object):
    """
    Training configuration container.
    """

    def __init__(self):
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

        object.__setattr__(self, "used_parameters", set())
        object.__setattr__(self, "undefined_parameters", set())

    def __getattr__(self, key):
        self.used_parameters.add(key)
        if key not in self.attrs:
            self.undefined_parameters.add(key)
            return None
        else:
            return self.attrs[key]

    def __setattr__(self, key, value):
        self.attrs[key] = value

    def get(self, key, default=None):
        return getattr(self, key, default=default)

    def report(self):
        """
        Report usage of training parameters.
        """
        logging.info("Accessed parameters in training configurations:")
        for key in self.used_parameters:
            logging.info(" - %s %s" % (key, "(undefined)" if key in self.undefined_parameters else ""))
