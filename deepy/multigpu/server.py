#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import sys, os
import numpy as np
import logging as loggers
from threading import Lock
logging = loggers.getLogger("ScheduledTrainingServer")
loggers.basicConfig(level=loggers.INFO)

from platoon.channel import Controller
from argparse import ArgumentParser


CONTROLLER_PORT = 5567

class ScheduledTrainingServer(Controller):
    """
    This multi-process controller implements patience-based early-stopping SGD
    """

    def __init__(self, port=CONTROLLER_PORT, easgd_alpha=0.5,
                 # Following arguments can be received from workers
                 start_halving_at=6, end_at=10, step_len=10,
                 valid_freq = 1500,
                 learning_rate = 0.1):
        """
        Initialize the controller.

        Args:
            port (int): batches in one training step
            easgd_alpha (float)
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
        self._training_names = []
        self._evaluation_names = []
        self._best_valid_cost = sys.float_info.max
        self._lock = Lock()

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

    def get_monitor_string(self, costs):
        return " ".join(["{}={:.2f}".format(n, c) for (n, c) in costs])


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
                if self._train_costs:
                    with self._lock:
                        sys.stdout.write("\r")
                        sys.stdout.flush()
                        mean_costs = []
                        for i in range(len(self._training_names)):
                            mean_costs.append(np.mean([c[i] for c in self._train_costs]))
                        logging.info("train   (epoch={:2d}) {}".format(
                            self.epoch,
                            self.get_monitor_string(zip(self._training_names, mean_costs)))
                        )
                response = {'eval': None, 'best_valid_cost': self._best_valid_cost}
                self._evaluating = True
            else:
                # Continue training
                if worker_id not in self.prepared_worker_pool:
                    response = {"sync_hyperparams": self.feed_hyperparams()}
                    self.prepared_worker_pool.add(worker_id)
                elif self._iters_from_last_valid >= self._valid_freq:
                    response = {'valid': None, 'best_valid_cost': self._best_valid_cost}
                    self._iters_from_last_valid = 0
                else:
                    response = {"train": self.feed_batches()}
        elif 'eval_done' in req:
            with self._lock:
                self._evaluating = False
                sys.stdout.write("\r")
                sys.stdout.flush()
                if 'test_costs' in req and req['test_costs']:
                    logging.info("test    (epoch={:2d}) {}".format(
                        self.epoch,
                        self.get_monitor_string(req['test_costs']))
                    )
                if 'valid_costs' in req and req['test_costs']:
                    valid_J = req['valid_costs'][0][1]
                    if valid_J < self._best_valid_cost:
                        self._best_valid_cost = valid_J
                        star_str = "*"
                    else:
                        star_str = ""
                    logging.info("valid   (epoch={:2d}) {} {}".format(
                        self.epoch,
                        self.get_monitor_string(req['valid_costs']),
                        star_str))
                    if star_str and 'auto_save' in req and req['auto_save']:
                        logging.info("(worker {}) save the model to {}".format(
                            worker_id,
                            req['auto_save']
                        ))
                continue_training = self.prepare_epoch()
                if not continue_training:
                    self._done = True
                    logging.info("training time {:.4f}s".format(time.time() - self.start_time))
                    response = "stop"
        elif 'valid_done' in req:
            with self._lock:
                sys.stdout.write("\r")
                sys.stdout.flush()
                if 'valid_costs' in req:
                    valid_J = req['valid_costs'][0][1]
                    if valid_J < self._best_valid_cost:
                        self._best_valid_cost = valid_J
                        star_str = "*"
                    else:
                        star_str = ""
                    logging.info("valid   ( dryrun ) {} {}".format(
                        self.get_monitor_string(req['valid_costs']),
                        star_str
                    ))
                    if star_str and 'auto_save' in req and req['auto_save']:
                        logging.info("(worker {}) save the model to {}".format(
                            worker_id,
                            req['auto_save']
                        ))
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
        elif 'init_schedule' in req:
            with self._lock:
                sys.stdout.write("\r")
                sys.stdout.flush()
                logging.info("worker {} connected".format(worker_id))
                if self.epoch == 0:
                    schedule_params = req['init_schedule']
                    sch_str = " ".join("{}={}".format(a, b) for (a, b) in schedule_params.items())
                    logging.info("initialize the schedule with {}".format(sch_str))
                    for key, val in schedule_params.items():
                        if not val: continue
                        if key == 'learning_rate':
                            self._lr = val
                        elif key == 'start_halving_at':
                            self.epoch_start_halving = val
                        elif key == 'end_at':
                            self.end_at = val
                        elif key == 'step_len':
                            self.step_len = val
                        elif key == 'valid_freq':
                            self._valid_freq = val

        elif 'set_names' in req:
            self._training_names = req['training_names']
            self._evaluation_names = req['evaluation_names']


        return response

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("--port", type=int, default=5567)
    ap.add_argument("--easgd_alpha", type=float, default=0.5)
    args = ap.parse_args()

    server = ScheduledTrainingServer(
        port=args.port,
        easgd_alpha=args.easgd_alpha)
    server.serve()
