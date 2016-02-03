#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
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
        thread.daemon = True
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
        try:
            thread._stop()
        except:
            pass
        return self.process.returncode

ERROR_KEYWORD = "Traceback (most recent call last)"

def test_experiment(path, timeout=60):
    path = os.path.join("experiments", path)
    executor = ParallelExecutor("python %s" % path)
    executor.run(timeout=timeout)
    if executor.stderr_output == None:
        logging.info("stderr is none")
        return
    if executor.stdout_output:
        logging.info(executor.stdout_output)
    if executor.stderr_output:
        logging.info(executor.stderr_output)
    if ERROR_KEYWORD in executor.stderr_output:
        logging.info("------ Error was found ------")
        logging.info(executor.stderr_output)
        logging.info("-----------------------------")
    else:
        logging.info("------ No error was found ------")
    assert ERROR_KEYWORD not in executor.stderr_output


if __name__ == '__main__':
    # Mnist tasks
    test_experiment("mnist/mlp_dropout.py", timeout=30)
    test_experiment("mnist/deep_convolution.py", timeout=30)
    # Auto-encoders
    test_experiment("auto_encoders/mnist_auto_encoder.py", timeout=60)
    test_experiment("auto_encoders/rnn_auto_encoder.py", timeout=120)
    # Highway networks
    test_experiment("highway_networks/mnist_highway.py", timeout=120)
    # TODO: recurrent neural networks
    # test_experiment("lm/baseline_rnnlm.py", timeout=180)
    # Scipy trainers
    test_experiment("scipy_training/mnist_cg.py", timeout=180)
    # Tutorials
    test_experiment("tutorials/tutorial1.py", timeout=120)
    # test_experiment("tutorials/tutorial2.py", timeout=120)
    # test_experiment("tutorials/tutorial3.py", timeout=120)
    sys.exit(0)
