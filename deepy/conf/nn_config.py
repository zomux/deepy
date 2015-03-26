#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging as loggers
logging = loggers.getLogger(__name__)

class NetworkConfig(object):

    def __init__(self, input_size):
        """
        Create a config for neural network
        """
        object.__setattr__(self, "attrs", {
            "input_size": input_size,
            "layers": [],
            "input_noise": 0.,
            "input_dropouts": 0.,
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
        key = getattr(self, key)
        if key != None:
            return key
        else:
            return default

    def report(self):
        """
        Report usage of training parameters.
        """
        logging.info("Accessed parameters in network configurations:")
        for key in self.used_parameters:
            logging.info(" - %s %s" % (key, "(undefined)" if key in self.undefined_parameters else ""))