#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import datetime
import logging as loggers
import deepy
logging = loggers.getLogger(__name__)

class TrainLogger(object):

    def __init__(self):
        self.log_pool = []

    def load(self, model_path):
        log_path = self._log_path(model_path)
        if os.path.exists(log_path):
            logging.info("Load training log from %s" % log_path)
            for line in open(log_path).xreadlines():
                self.log_pool.append(line.strip())

    def record(self, line):
        time_mark = datetime.datetime.now().strftime("[%Y/%m/%d %H:%M:%S] ")
        self.log_pool.append(time_mark + line)

    def save(self, model_path):
        log_path = self._log_path(model_path)
        logging.info("Save training log to %s" % log_path)
        with open(log_path, "w") as outf:
            outf.write("# deepy version: %s\n" % deepy.__version__)
            for line in self.log_pool:
                outf.write(line + "\n")

    def _log_path(self, model_path):
        log_path = model_path.rsplit(".", 1)[0] + ".log"
        return log_path