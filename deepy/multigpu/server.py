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
from collections import defaultdict


CONTROLLER_PORT = 5567

class ScheduledTrainingServer(Controller):
    """
    This multi-process controller implements patience-based early-stopping SGD
    """

    def __init__(self, port=CONTROLLER_PORT, device_num=1,
                 # Following arguments can be received from workers
                 anneal_rule='simple',
                 start_anneal_at=6, anneal_freq=1,
                 anneal_factor=0.5, anneal_patience=1,
                 anneal_times = 4,
                 end_at=10,
                 pack_size=5,
                 valid_freq=1500, learning_rate=0.1, log_path=None):
        """
        Initialize the controller.

        Args:
            port (int): batches in one training step
            easgd_alpha (float)
            anneal_rule: simple or schedule or patience
        """

        Controller.__init__(self, port)
        self.start_anneal_at = start_anneal_at
        self.end_at = end_at
        self.pack_size = pack_size
        self.start_time = None
        self.rand = np.random.RandomState(3)
        self.epoch = 0
        self.device_num = device_num
        self._current_iter = 0
        self._iters_from_last_valid = 0
        self._evaluating = False
        self._valid_freq = valid_freq
        self._anneal_freq = anneal_freq
        self._anneal_rule = anneal_rule
        self._anneal_factor = anneal_factor
        self._anneal_patience = anneal_patience
        self._anneal_times = anneal_times
        self._annealed_times = 0
        self._done = False
        self._lr = learning_rate
        self._training_names = []
        self._evaluation_names = []
        self._best_valid_cost = sys.float_info.max
        self._server_lock = Lock()

        self.num_train_batches = 0
        self.batch_pool = []
        self._train_costs = []
        self._epoch_start_time = None
        self._n_failed_valid = 0
        self.msgbox = defaultdict(list)
        self.prepared_worker_pool = set()
        self.log_text = ""
        self.log_file = open(log_path, "w") if log_path else None
        if log_path:
            logging.info("write logs into {}".format(log_path))
        logging.info("multi-gpu server is listening port {} for {} devices".format(port, self.device_num))

    def serve(self):
        from platoon.util import PlatoonError
        import zmq
        try:  # spin spin spin
            self._success = 2
            while True:  # spin while we have still children to watch for
                try:
                    query = self.csocket.recv_json(flags=zmq.NOBLOCK)
                except zmq.Again:  # if a query has not happened, try again
                    continue
                except zmq.ZMQError as exc:
                    raise PlatoonError("while receiving using ZMQ socket", exc)
                # try default interface, it may raise PlatoonError
                # print(query['req'])
                response = self._handle_base_control(query['req'],
                                                     query['worker_id'],
                                                     query['req_info'])
                if response is None:
                    response = self.handle_control(query['req'],
                                                   query['worker_id'],
                                                   query['req_info'])
                try:
                    self.csocket.send_json(response)
                except zmq.ZMQError as exc:
                    raise PlatoonError("while sending using ZMQ socket", exc)
        except PlatoonError as exc:  # if platoon fails kill all children workers
            print(exc)
            self._clean()
        # except Exception as exc:
        #     print(PlatoonError("Unexpected exception", exc))
        #     self._clean()
        finally:
            # Close sockets and unlink for shared memory
            self._close()
        return self._success

    def _get_platoon_info(self, req_info):
        if req_info['device'] not in self._devices:
            self._devices.append(req_info['device'])
        self._local_size = self.device_num
        self._global_size = self._local_size
        first = self._is_worker_first(self._get_platoon_info_count)  # See :meth:`_is_worker_first`
        if first:
            self._local_id = "multigpu-" + req_info['local_id']
        response = dict()
        response['local_id'] = self._local_id
        response['local_size'] = self._local_size
        local_rank = self._devices.index(req_info['device'])
        response['local_rank'] = local_rank
        response['multinode'] = self._multinode
        response['global_size'] = self._global_size
        if self._multinode:
            response['global_rank'] = sum(self._all_local_size[:self._region_rank]) + local_rank
        else:
            response['global_rank'] = local_rank
        return response

    def _is_worker_first(self, counter):
        counter[0] = (counter[0] + 1)
        if counter[0] == 1:
            return True
        return False

    def log(self, msg):
        logging.info(msg)
        self.log_text += msg + "\n"
        if self.log_file:
            self.log_file.write(msg + "\n")

    def prepare_epoch(self):
        """
        Prepare for one epoch.
        Returns:
            bool: False if to stop the training.
        """
        self.epoch += 1
        if self._anneal_rule == 'schedule':
            if self.epoch >= self.start_anneal_at and ((self.epoch - self.start_anneal_at) % self._anneal_freq == 0):
                self._lr *= self._anneal_factor
        self._current_iter = 0
        self._iters_from_last_valid = 0
        self._train_costs = []
        self.prepared_worker_pool.clear()
        self.batch_pool = range(self.num_train_batches)
        self.rand.shuffle(self.batch_pool)
        # Ensure that # pool % (dev * pack) == 0
        round_size = self.device_num * self.pack_size
        self.batch_pool = self.batch_pool[:(len(self.batch_pool) // round_size) * round_size]
        if self.epoch > self.end_at:
            self.log("Training is done, wait all workers to stop")
            return False
        else:
            self.log("start epoch {} with lr={}".format(self.epoch, self._lr))
            return True

    def feed_batches(self):
        if not self.batch_pool:
            return None
        else:
            batches = self.batch_pool[:self.pack_size]
            self.batch_pool = self.batch_pool[self.pack_size:]
            self._current_iter += len(batches)
            self._iters_from_last_valid += len(batches)
            return batches


    def feed_hyperparams(self):
        retval = {
            "epoch": self.epoch,
            "learning_rate": self._lr
        }
        return retval

    def get_monitor_string(self, costs):
        return " ".join(["{}={:.2f}".format(n, c) for (n, c) in costs])


    def handle_control(self, req, worker_id, req_info):
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
        if worker_id not in self.msgbox:
            self.msgbox[worker_id] = []
        response = ""

        if req == 'next':
            if self.msgbox[worker_id]:
              response = self.msgbox[worker_id].pop()
            elif self.num_train_batches == 0:
                response = "get_num_batches"
            elif self._evaluating:
                response = 'wait'
            elif not self.batch_pool and self._done:
                response = "stop"
                self.worker_is_done(worker_id)
            elif not self.batch_pool:
                # End of one iter
                if self._train_costs:
                    with self._server_lock:
                        sys.stdout.write("\r")
                        sys.stdout.flush()
                        mean_costs = []
                        for i in range(len(self._training_names)):
                            mean_costs.append(np.mean([c[i] for c in self._train_costs]))
                        self.log("train   (epoch={:2d}) {}".format(
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
            with self._server_lock:
                self._evaluating = False
                sys.stdout.write("\r")
                sys.stdout.flush()
                if 'test_costs' in req and req['test_costs']:
                    self.log("test    (epoch={:2d}) {} (worker {})".format(
                        self.epoch,
                        self.get_monitor_string(req['test_costs']),
                        worker_id)
                    )
                if 'valid_costs' in req and req['test_costs'] and False:
                    valid_J = req['valid_costs'][0][1]
                    if valid_J <= self._best_valid_cost:
                        self._best_valid_cost = valid_J
                        star_str = "*"
                    else:
                        star_str = ""
                        self.log("valid   (epoch={:2d}) {} {} (worker {})".format(
                        self.epoch,
                        self.get_monitor_string(req['valid_costs']),
                        star_str,
                        worker_id))
                continue_training = self.prepare_epoch()
                self._epoch_start_time = time.time()
                if not continue_training:
                    self._done = True
                    self.log("training time {:.4f}s".format(time.time() - self.start_time))
                    response = "stop"
        elif 'valid_done' in req:
            with self._server_lock:
                sys.stdout.write("\r")
                sys.stdout.flush()
                if 'valid_costs' in req:
                    valid_J = req['valid_costs'][0][1]
                    if valid_J <= self._best_valid_cost:
                        self._best_valid_cost = valid_J
                        star_str = "*"
                        self._n_failed_valid = 0
                    else:
                        star_str = ""
                        self._n_failed_valid += 1
                    self.log("valid   ( dryrun ) {} {} (worker {})".format(
                        self.get_monitor_string(req['valid_costs']),
                        star_str,
                        worker_id
                    ))
                if self._anneal_rule == 'patience' and self._n_failed_valid >= self._anneal_patience:
                    # Patience running out
                    if self._annealed_times >= self._anneal_times:
                        self._done = True
                        self.log("stop after annealing {} times".format(self._annealed_times))
                        self.log("training time {:.4f}s".format(time.time() - self.start_time))
                        # Clean up batch_pool
                        round_size = self.pack_size * self.device_num
                        self.batch_pool = self.batch_pool[:len(self.batch_pool) % round_size]
                    else:
                        self._lr *= self._anneal_factor
                        self._n_failed_valid = 0
                        self._annealed_times += 1
                        self.log("annealing learning rate to {}".format(self._lr))
                        for wid in self.msgbox:
                            self.msgbox[wid].append({"sync_hyperparams": self.feed_hyperparams()})
        elif 'train_done' in req:
            costs = req['costs']
            self._train_costs.append(costs)
            sys.stdout.write("\x1b[2K\r> %d%% | J=%.2f | %.1f batch/s" % (
                self._current_iter * 100 / self.num_train_batches,
                costs[0], float(len(self._train_costs) * self.pack_size) / (time.time() - self._epoch_start_time)))
            sys.stdout.flush()
        elif 'get_num_batches_done' in req:
            self.num_train_batches = req['get_num_batches_done']
        elif 'sync_hyperparams' in req:
            response = {"sync_hyperparams": self.feed_hyperparams()}
        elif req == "get_log_text":
            response = self.log_text
        elif 'init_schedule' in req:
            with self._server_lock:
                sys.stdout.write("\r")
                sys.stdout.flush()
                self.log("worker {} connected".format(worker_id))
                if self.epoch == 0:
                    schedule_params = req['init_schedule']
                    sch_str = " ".join("{}={}".format(a, b) for (a, b) in schedule_params.items())
                    self.log("initialize the schedule with {}".format(sch_str))
                    for key, val in schedule_params.items():
                        if not val: continue
                        if key == 'learning_rate':
                            self._lr = val
                        elif key == 'start_anneal_at':
                            self.start_anneal_at = val
                        elif key == 'anneal_freq':
                            self._anneal_freq = val
                        elif key == 'end_at':
                            self.end_at = val
                        elif key == 'pack_size':
                            self.pack_size = val
                        elif key == 'valid_freq':
                            self._valid_freq = val
                        elif key == 'anneal_factor':
                            self._anneal_factor = val
                        elif key == 'anneal_rule':
                            self._anneal_rule = val
                        elif key == 'anneal_patience':
                            self._anneal_patience = val
                        elif key == 'anneal_times':
                            self._anneal_times = val

        elif 'set_names' in req:
            self._training_names = req['training_names']
            self._evaluation_names = req['evaluation_names']


        return response

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("--port", type=int, default=5567)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--log", type=str, default=None)
    args = ap.parse_args()

    server = ScheduledTrainingServer(
        port=args.port,
        device_num=args.devices,
        log_path=args.log
    )

    server.serve()
