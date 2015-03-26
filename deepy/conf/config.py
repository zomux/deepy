#!/usr/bin/env python
# -*- coding: utf-8 -*-


class GeneralConfig(object):

    def __init__(self, logger=None):
        """
        Create a general config
        """
        object.__setattr__(self, "attrs", {})
        object.__setattr__(self, "used_parameters", set())
        object.__setattr__(self, "undefined_parameters", set())
        object.__setattr__(self, "logger", logger)

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

    def merge(self, config):
        for k in config.attrs:
            setattr(self, k, getattr(config, k))
        return self

    def report(self):
        """
        Report usage of training parameters.
        """
        if self.logger:
            self.logger.info("accessed parameters:")
            for key in self.used_parameters:
                self.logger.info(" - %s %s" % (key, "(undefined)" if key in self.undefined_parameters else ""))
