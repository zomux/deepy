#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict, defaultdict
import logging
import numpy as np

class TrainingController(object):
    """
    Abstract class of training controllers.
    """

    def bind(self, trainer):
        """
        :type trainer: deepy.trainers.base.NeuralTrainer
        """
        self._trainer = trainer

    def invoke(self):
        """
        Return True to exit training.
        """
        return False

class TrainingValidator(TrainingController):
    """
    A validator that allows validating the model with another graph.
    """

    def __init__(self, valid_model=None, data_split='valid', freq=1500, save_path=None, criteria='cost',
                 smaller_is_better=True, annealing=False, anneal_times=3, anneal_factor=0.5):
        """
        Initialize the training validator.
        """
        self._model = valid_model
        self._data_split = data_split
        self._freq = freq
        self._save_path = save_path
        self._criteria = criteria
        self._smaller_is_better = smaller_is_better
        self._best_criteria = None
        self._counter = 0
        self._annealing = annealing
        self._anneal_times = anneal_times
        self._annealed_times = 0
        self._anneal_factor = 0.5

    def compare(self, cost_map):
        """
        Compare to previous records and return whether the given cost is a new best.
        :return: True if the given cost is a new best
        """
        cri_val = cost_map[self._criteria]
        if self._best_criteria is None:
            self._best_criteria = cri_val
            return True
        else:
            if self._smaller_is_better and cri_val < self._best_criteria:
                self._best_criteria = cri_val
                return True
            elif not self._smaller_is_better and cri_val > self._best_criteria:
                self._best_criteria = cri_val
                return True
            else:
                return False

    def compute(self, *x):
        """
        Compute with the validation model given data x.
        """
        return self._model.compute(*x)

    def _extract_costs(self, vars):
        ret_map = OrderedDict()
        sub_costs = OrderedDict()
        for k, val in vars.items():
            if val.ndim == 0:
                if k == self._criteria:
                    ret_map[k] = val
                else:
                    sub_costs[k] = val
        ret_map.update(sub_costs)
        return ret_map

    def run(self, data_x):
        """
        Run the model with validation data and return costs.
        """
        output_vars = self.compute(*data_x)
        return self._extract_costs(output_vars)

    def invoke(self):
        """
        This function will be called after each iteration.
        """
        from deepy import runtime, FLOATX
        self._counter += 1
        if self._counter % self._freq == 0:
            cnt = 0.
            sum_map = defaultdict(float)
            runtime.switch_training(False)
            for x in self._trainer.get_data(self._data_split):
                val_map = self.run(x)
                if not isinstance(val_map, dict):
                    raise Exception("Monitor.run must return a dict.")
                for k, val in val_map.items():
                    sum_map[k] += val
                cnt += 1
            
            for k in sum_map:
                sum_map[k] /= cnt
            new_best = self.compare(sum_map)
            self._trainer.report(sum_map, self._data_split, new_best=new_best)
            if new_best:
                self._trainer.save_checkpoint(self._save_path)
            if self._annealing and not new_best:
                lr = self._trainer.config.learning_rate
                if self._annealed_times >= self._anneal_times:
                    logging.info("ending")
                    self._trainer.exit()
                else:
                    lr.set_value(
                        np.array(lr.get_value() * self._anneal_factor, dtype=FLOATX))
                    self._annealed_times += 1
                    logging.info("annealed learning rate to %f" % lr.get_value())
