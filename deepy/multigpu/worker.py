#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os
from collections import OrderedDict
import numpy as np

from platoon.channel import Worker
from platoon.param_sync import EASGD
from deepy.trainers import GeneralNeuralTrainer

import logging


class MultiGPUTrainer(GeneralNeuralTrainer):
    """
    General neural network trainer.
    """
    def __init__(self,
                 network, config=None, method=None,
                 server_port=5567,
                 start_halving_at=6, end_at=10, step_len=10,
                 valid_freq=1500, learning_rate=None
                 ):
        super(MultiGPUTrainer, self).__init__(network, config, method)
        self._report_time = False
        self._port = server_port
        self.logger = logging.getLogger('MultiGPUTrainingWorker')
        self.epoch = 0
        if not learning_rate:
            learning_rate = float(self.config.learning_rate.get_value())
        self._schedule_params = {
            'learning_rate': learning_rate,
            'start_halving_at': start_halving_at,
            'end_at': end_at,
            'step_len': step_len,
            'valid_freq': valid_freq
        }

    def create_param_map(self):
        param_map = OrderedDict()
        for i, param in enumerate(self.training_params()):
            param_map["param_{}".format(i)] = param
        return param_map

    def sync_hyperparams(self, param_map):
        self.logger.info("(proc {}) sync hyperparameters".format(os.getpid()))
        if 'epoch' in param_map:
            self.epoch = param_map['epoch']
        if 'learning_rate' in param_map:
            self.config.learning_rate.set_value(param_map['learning_rate'])

    def fix_costs(self):
        self.last_run_costs = [(a, float(b)) for (a,b) in self.last_run_costs]

    def train(self, train_set, valid_set=None, test_set=None, train_size=None):
        """
        Train the model in multi-GPU environment.
        """
        server_port = self._port
        param_map = self.create_param_map()
        # Initialize the worker
        worker = Worker(control_port=server_port)
        if self.config.learning_rate:
            worker.send_req({'init_schedule': self._schedule_params})
        self.sync_hyperparams(worker.send_req('sync_hyperparams')['sync_hyperparams'])
        easgd_alpha = worker.send_req('get_easgd_alpha')
        worker.init_shared_params(param_map.values(), param_sync_rule=EASGD(easgd_alpha))
        worker.copy_to_local()
        worker.send_req({
            "set_names": None,
            "training_names": self.training_names,
            "evaluation_names": self.evaluation_names
        })
        # Load all training batches, consume vast memory here
        self.logger.info("started process {}".format(os.getpid()))
        self.logger.info("(proc {}) load training data".format(os.getpid()))
        train_batches = list(train_set)
        network_callback = bool(self.network.training_callbacks)
        trainer_callback = bool(self._iter_callbacks)
        while True:
            resp = worker.send_req('next')
            if resp == 'stop':
                break
            elif resp == 'wait':
                time.sleep(1)
            elif resp == 'get_num_batches':
                worker.send_req({'get_num_batches_done': len(train_batches)})
            elif 'eval' in resp:
                self.best_cost = resp['best_valid_cost']
                worker.copy_to_local()
                valid_costs = None
                test_costs = None
                if valid_set:
                    self._run_valid(self.epoch, valid_set)
                    self.fix_costs()
                    valid_costs = self.last_run_costs
                if test_set:
                    self._run_test(self.epoch, test_set)
                    self.fix_costs()
                    test_costs = self.last_run_costs
                worker.send_req({
                    "eval_done": None,
                    "valid_costs": valid_costs,
                    "test_costs": test_costs,
                    "auto_save": self.config.auto_save
                })
            elif 'valid' in resp:
                self.best_cost = resp['best_valid_cost']
                worker.copy_to_local()
                if valid_set:
                    self._run_valid(self.epoch, valid_set, dry_run=True)
                    self.fix_costs()
                worker.send_req({
                    "valid_done": None,
                    "valid_costs": self.last_run_costs,
                    "auto_save": self.config.auto_save
                })
            elif 'train' in resp:
                batch_ids = resp['train']
                batch_costs = [[] for _ in self.training_names]
                for batch_id in batch_ids:
                    x = train_batches[batch_id]
                    cost_x = self.learn(*x)
                    for i, cost in enumerate(cost_x):
                        batch_costs[i].append(cost)
                    self.last_cost = cost_x[0]
                if network_callback:
                    self.network.training_callback()
                if trainer_callback:
                    for func in self._iter_callbacks:
                        func(self)
                worker.sync_params(synchronous=True)
                worker.send_req({'train_done': None, 'costs': [float(np.mean(c)) for c in batch_costs]})
            elif 'sync_hyperparams' in resp:
                self.sync_hyperparams(resp['sync_hyperparams'])
        worker.close()
        return []
