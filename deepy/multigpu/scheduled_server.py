#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import sys, os
import numpy as np
import logging as loggers
logging = loggers.getLogger("ScheduledTrainingServer")
loggers.basicConfig(level=loggers.INFO)

from platoon.channel import Controller
from argparse import ArgumentParser


CONTROLLER_PORT = 5567

class ScheduledTrainingServer(Controller):
    """
    This multi-process controller implements patience-based early-stopping SGD
    """

    def __init__(self, port=CONTROLLER_PORT, start_halving_at=5, end_at=10, step_len=10,
                 valid_freq = 1000,
                 learning_rate = 0.1,
                 easgd_alpha=0.5):
        """
        Initialize the controller.

        Args:
            step_len (int): batches in one training step
            config (dict)
        """

        Controller.__init__(self, port)
        self.epoch_start_halving = start_halving_at
        self.end_at = end_at
        self.step_len = step_len
        self.start_time = None
        self.rand = np.random.RandomState(3)
        self.epoch = 0
        self._current_iter = 0
        self._iters_from_last_valid = 0
        self._evaluating = False
        self._valid_freq = valid_freq
        self._done = False
        self._lr = learning_rate
        self._easgd_alpha = easgd_alpha

        self.num_train_batches = 0
        self.batch_pool = []
        self._train_costs = []
        self.prepared_worker_pool = set()
        logging.info("multi-gpu server is listening port {}".format(port))

    def prepare_epoch(self):
        """
        Prepare for one epoch.
        Returns:
            bool: False if to stop the training.
        """
        self.epoch += 1
        if self.epoch >= self.epoch_start_halving:
            self._lr *= 0.5
        self._current_iter = 0
        self._iters_from_last_valid = 0
        self._train_costs = []
        self.prepared_worker_pool.clear()
        self.batch_pool = range(self.num_train_batches)
        self.rand.shuffle(self.batch_pool)
        if self.epoch > self.end_at:
            logging.info("Training is done, wait all workers to stop")
            return False
        else:
            logging.info("start epoch {} with lr={}".format(self.epoch, self._lr))
            return True

    def feed_batches(self):
        if not self.batch_pool:
            return None
        else:
            batches = self.batch_pool[:self.step_len]
            self.batch_pool = self.batch_pool[self.step_len:]
            self._current_iter += len(batches)
            self._iters_from_last_valid += len(batches)
            return batches


    def feed_hyperparams(self):
        retval = {
            "epoch": self.epoch,
            "learning_rate": self._lr,
            "easgd_alpha": self._easgd_alpha
        }
        return retval


    def handle_control(self, req, worker_id):
        """
        Handles a control_request received from a worker.
        Returns:
            string or dict: response

            'stop' - the worker should quit
            'wait' - wait for 1 second
            'eval' - evaluate on valid and test set to start a new epoch
            'sync_hyperparams' - set learning rate
            'valid' - evaluate on valid and test set, then save the params
            'train' - train next batches
        """
        if self.start_time is None: self.start_time = time.time()
        response = ""

        if req == 'next':
            if self.num_train_batches == 0:
                response = "get_num_batches"
            elif self._done:
                response = "stop"
                self.worker_is_done(worker_id)
            elif self._evaluating:
                response = 'wait'
            elif not self.batch_pool:
                # End of one iter
                response = 'eval'
                self._evaluating = True
            else:
                # Continue training
                if worker_id not in self.prepared_worker_pool:
                    response = {"sync_hyperparams": self.feed_hyperparams()}
                    self.prepared_worker_pool.add(worker_id)
                elif self._iters_from_last_valid >= self._valid_freq:
                    response = 'valid'
                    self._iters_from_last_valid = 0
                else:
                    response = {"train": self.feed_batches()}
        elif 'eval_done' in req:
            messages = req['eval_done']
            self._evaluating = False
            sys.stdout.write("\r")
            sys.stdout.flush()
            for msg in messages:
                logging.info(msg)
            continue_training = self.prepare_epoch()
            if not continue_training:
                self._done = True
                logging.info("training time {:.4f}s".format(time.time() - self.start_time))
                response = "stop"
        elif 'valid_done' in req:
            messages = req['valid_done']
            sys.stdout.write("\r")
            sys.stdout.flush()
            for msg in messages:
                logging.info(msg)
        elif 'train_done' in req:
            costs = req['costs']
            self._train_costs.append(costs)
            sys.stdout.write("\x1b[2K\r> %d%% | J=%.2f" % (self._current_iter * 100 / self.num_train_batches,
                                                           costs[0]))
            sys.stdout.flush()
        elif 'get_num_batches_done' in req:
            self.num_train_batches = req['get_num_batches_done']
        elif 'get_easgd_alpha' in req:
            response = self._easgd_alpha
        elif 'sync_hyperparams' in req:
            response = {"sync_hyperparams": self.feed_hyperparams()}

        return response

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("--port", type=int, default=5567)
    ap.add_argument("--learning_rate", type=float, default=0.01)
    ap.add_argument("--start_halving_at", type=int, default=5)
    ap.add_argument("--end_at", type=int, default=10)
    ap.add_argument("--step_len", type=int, default=10)
    ap.add_argument("--valid_freq", type=int, default=1500)
    ap.add_argument("--easgd_alpha", type=float, default=0.5)
    args = ap.parse_args()

    server = ScheduledTrainingServer(
        port=args.port, learning_rate=args.learning_rate,
        start_halving_at=args.start_halving_at,
        end_at=args.end_at,
        step_len=args.step_len,
        valid_freq=args.valid_freq,
        easgd_alpha=args.easgd_alpha)
    server.serve()
