#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import os
import subprocess, threading
import logging as loggers
logging = loggers.getLogger(__name__)

class ParallelExecutor(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None
        self.stdout_output = ""
        self.stderr_output = ""

    def run(self, timeout):
        def target():
            logging.info(self.cmd)
            self.process = subprocess.Popen(self.cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.stdout_output, self.stderr_output = self.process.communicate()
            logging.info("Thread finished")

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            logging.info("Time out, terminating process")
            self.process.terminate()
            thread.join()
        return self.process.returncode

ERROR_KEYWORD = "Traceback (most recent call last)"

class ExperimentTest(unittest.TestCase):

    def _test(self, path, timeout=60):
        path = os.path.join("experiments", path)
        executor = ParallelExecutor("python %s" % path)
        executor.run(timeout=timeout)
        if ERROR_KEYWORD in executor.stderr_output or ERROR_KEYWORD in executor.stdout_output:
            logging.info("stdout:")
            logging.info(executor.stdout_output)
            logging.info("stderr:")
            logging.info(executor.stderr_output)
        self.assertTrue(ERROR_KEYWORD not in executor.stderr_output)
        self.assertTrue(ERROR_KEYWORD not in executor.stdout_output)

    # MNIST
    def test_mnist_mlp_dropout(self):
        self._test("mnist/mlp_dropout.py", timeout=30)

    def test_mnist_deep_conv(self):
        self._test("mnist/deep_convolution.py", timeout=30)

    # Language models
    def test_baseline_rnnlm(self):
        self._test("lm/baseline_rnnlm.py")

    # Highway networks
    def test_highway(self):
        self._test("highway_networks/mnist_highway.py")

