#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess, threading
import logging as loggers
import time
loggers.basicConfig(level=loggers.INFO)
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
            time.sleep(1)
            try:
                self.process.kill()
            except OSError:
                pass
            # thread.join()
        return self.process.returncode

ERROR_KEYWORD = "Traceback (most recent call last)"

def test_experiment(path, timeout=60):
    path = os.path.join("experiments", path)
    executor = ParallelExecutor("python %s" % path)
    executor.run(timeout=timeout)
    if executor.stderr_output == None:
        logging.info("stderr is none")
        return
    if ERROR_KEYWORD in executor.stderr_output:
        logging.info("stdout:")
        logging.info(executor.stdout_output)
        logging.info("stderr:")
        logging.info(executor.stderr_output)
    assert ERROR_KEYWORD not in executor.stderr_output


if __name__ == '__main__':
    # MNIST
    test_experiment("mnist/mlp_dropout.py", timeout=30)
    test_experiment("mnist/deep_convolution.py", timeout=30)
    # LMs
    test_experiment("lm/baseline_rnnlm.py")
    # Highway networks
    test_experiment("highway_networks/mnist_highway.py")
