#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os
from collections import OrderedDict
import numpy as np
import theano

from deepy.trainers import GeneralNeuralTrainer
from deepy.utils.timeout import TimeoutError, timeout
from deepy.core import runtime

import logging

class MultiGPUTrainer(GeneralNeuralTrainer):
    """
    General neural network trainer.
    """

    def __init__(self,
                 network, config=None, method='sgd',
                 server_port=5567,
                 anneal_rule='simple',
                 start_anneal_at=6, anneal_freq=1,
                 anneal_factor=0.5, anneal_patience=1,
                 anneal_times=4,
                 end_at=10,
                 pack_size=5,
                 valid_freq=1500, learning_rate=None,
                 type="average", reload_on_anneal=False,
                 ):
        super(MultiGPUTrainer, self).__init__(network, method, config)
        self._report_time = False
        self._port = server_port
        self.logger = logging.getLogger('MultiGPUTrainingWorker')
        self.epoch = 0
        self._type = type
        self._reload_on_anneal = reload_on_anneal
        if not learning_rate:
            learning_rate = float(self.config.learning_rate.get_value())
        self._optimize_func = None
        self._schedule_params = {
            'learning_rate': learning_rate,
            'start_anneal_at': start_anneal_at,
            'end_at': end_at,
            'pack_size': pack_size,
            'valid_freq': valid_freq,
            'anneal_freq': anneal_freq,
            'anneal_rule': anneal_rule,
            'anneal_factor': anneal_factor,
            'anneal_patience': anneal_patience,
            'anneal_times': anneal_times
        }
    
    def build_gradient_caches(self):
        """
        Build shared variables contaning gradients.
        """
        self.gradient_caches = []
        params = self.training_params()
        for i, param in enumerate(params):
            val = param.get_value()
            grad_cache = theano.shared(val * 0, name="gc_{}".format(i))
            self.gradient_caches.append(grad_cache)
            
    def _learning_updates(self):
        """
        Updating gradient caches.
        """
        params = self.training_params()
        gradients = self.get_gradients(params)
        gc_updates = []
        for grad, gc in zip(gradients, self.gradient_caches):
            gc_updates.append((gc, grad))
        return gc_updates
    
    def learning_function(self):
        """
        Get the learning function.
        :param func:
        :return:
        """
        network_updates = list(self.network.updates) + list(self.network.training_updates)
        learning_updates = list(self._learning_updates())
        update_list = network_updates + learning_updates

        logging.info("network updates: %s" % " ".join(map(str, [x[0] for x in network_updates])))
        logging.info("gradient updates: %s" % " ".join(map(str, [x[0] for x in learning_updates])))

        variables = self.network.input_variables + self.network.target_variables
        givens = None
        return theano.function(
            variables,
            map(lambda v: theano.Out(v, borrow=True), self.training_variables),
            updates=update_list, allow_input_downcast=True,
            mode=self.config.get("theano_mode", None),
            givens=givens)
    
    def optimization_function(self):
        """
        Compile the optimization function.
        """
        params = self.training_params()
        opt_updates = self.optimization_updates(params, self.gradient_caches)
        logging.info("optimization updates: %s" % " ".join(map(str, [x[0] for x in opt_updates])))
        return theano.function(
            [],
            None,
            updates=opt_updates, allow_input_downcast=True,
            mode=self.config.get("theano_mode", None))
    
    def update_params(self):
        if not self._optimize_func:
            start_time = time.time()
            logging.info('compiling optimization function')
            self._optimize_func = self.optimization_function()
            compile_time = time.time() - start_time
            logging.info("took {} seconds to compile".format(int(compile_time)))
        return self._optimize_func()

    def check_param_hash(self):
        params = self.training_params()
        hash_str = " ".join(["{:.2f}".format(p.get_value().max()) for p in params])
        logging.info("param hash: {}".format(hash_str))

    def sync_hyperparams(self, param_map):
        if abs(self.config.learning_rate.get_value() -  param_map['learning_rate']) < 1e-08:
            self.logger.info(
                "(proc {}) set learning rate to {}".format(
                    os.getpid(), param_map['learning_rate']))
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
        from platoon.channel import Worker
        from platoon.training.global_dynamics import AverageSGD
        server_port = self._port
        self.check_param_hash()
        # Initialize the worker
        worker = Worker(control_port=server_port)
        if self.config.learning_rate:
            worker.send_req({'init_schedule': self._schedule_params})
        self.sync_hyperparams(worker.send_req('sync_hyperparams')['sync_hyperparams'])
        self.logger.info("Synchronizing type: {}".format(self._type))
        if self._type != "average":
            raise NotImplementedError
        # Build gradient sync rule
        self.build_gradient_caches()
        training_cnt = 0
        sync_rule = AverageSGD(worker)
        sync_rule.make_rule(self.gradient_caches)
        param_sync_rule = AverageSGD(worker)
        param_sync_rule.make_rule(self.network.all_parameters)
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
        trainer_callback = bool(self._iter_controllers)
        # Start from valid, so the performance when a worked join can be known
        if valid_set:
            self._run_valid(self.epoch, valid_set, dry_run=True)
            self.fix_costs()
        # Do not send results
            # worker.send_req({
            #     "valid_done": None,
            #     "valid_costs": self.last_run_costs,
            #     "auto_save": self.config.auto_save
            # })
        # Pre-sync params
        self.logger.info("pre-sync parameters ...")
        param_sync_rule()
        self.check_param_hash()
        # Begin the loop
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
                valid_costs = None
                test_costs = None
                if valid_set and False:
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
                if valid_set:
                    log_resp = worker.send_req('get_log_text')
                    self.network.train_logger.log_pool = log_resp.split("\n")
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
                runtime.switch_training(True)
                for batch_id in batch_ids:
                    x = train_batches[batch_id]
                    cost_x = self.learn(*x)
                    for i, cost in enumerate(cost_x):
                        batch_costs[i].append(cost)
                    self.last_cost = cost_x[0]
                    try:
                        with timeout(seconds=10):
                            sync_rule()
                    except TimeoutError:
                        runtime.switch_training(False)
                        break
                    self.update_params()
                runtime.switch_training(False)
                if network_callback:
                    self.network.training_callback()
                if trainer_callback:
                    for func in self._iter_controllers:
                        func(self)
                training_cnt += 1
                if training_cnt >= 100:
                    param_sync_rule()
                    training_cnt = 0
                worker.send_req({'train_done': None, 'costs': [float(np.mean(c)) for c in batch_costs]})
            elif 'sync_hyperparams' in resp:
                self.sync_hyperparams(resp['sync_hyperparams'])
                if self._reload_on_anneal and "reload" in resp and resp["reload"] and os.path.exists(self.config.auto_save):
                    self.logger.info("reloading parameters ...")
                    self.load_params(self.config.auto_save)
                
        worker.close()
        return []
