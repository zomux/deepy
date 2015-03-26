#!/usr/bin/env python
# -*- coding: utf-8 -*-

from config import GeneralConfig
import logging as loggers
logging = loggers.getLogger(__name__)

class NetworkConfig(GeneralConfig):
    """
    Network configuration container.
    """
    def __init__(self, input_size):
        super(NetworkConfig, self).__init__(logger=logging)
        object.__setattr__(self, "attrs", {
            "input_size": input_size,
            "layers": [],
            "input_noise": 0.,
            "input_dropouts": 0.,
        })