#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

class Timer(object):
    """
    A timer for measure runing time.
    """

    def __init__(self):
        self.start_time = time.time()
        self.end_time = None

    def end(self):
        """
        Stop the timer.
        """
        self.end_time = time.time()

    def report(self):
        """
        Report elapsed time.
        """
        if not self.end_time:
            self.end()
        print "Time:", ((self.end_time - self.start_time )/ 60), "mins"
