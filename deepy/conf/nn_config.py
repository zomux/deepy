#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy.utils import UniformInitializer
from config import GeneralConfig

import logging as loggers
logging = loggers.getLogger(__name__)

class NetworkConfig(GeneralConfig):
    """
    Network configuration container.
    """
    def __init__(self):
        super(NetworkConfig, self).__init__(logger=logging)
        object.__setattr__(self, "attrs", dict())