#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy.utils import UniformInitializer
from config import GeneralConfig

import logging as loggers
logging = loggers.getLogger(__name__)

DEFAULT_NETWORK_SETTING = {}

class NetworkConfig(GeneralConfig):
    """
    Network configuration container.
    """
    def __init__(self, settingMap=None):
        super(NetworkConfig, self).__init__(logger=logging)

        settings = DEFAULT_NETWORK_SETTING
        if isinstance(settingMap, dict):
            settings.update(settingMap)

        for key, value in settings.items():
            self.attrs[key] = value