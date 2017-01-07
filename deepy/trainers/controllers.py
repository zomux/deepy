#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict, defaultdict

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

class TrainingMonitor(TrainingController):
    """
    A monitor called between iterations.
    """

    def __init__(self, valid_model, data_split='valid', freq=1500, save_path=None, criteria='cost',
                 smaller_is_better=True):
        """
        Initialize the training monitor.
        """
        self._model = valid_model
        self._data_split = data_split
        self._freq = freq
        self._save_path = None
        self._criteria = criteria
        self._smaller_is_better = smaller_is_better
        self._best_criteria = None
        self._counter = 0

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
        self._counter += 1
        if self._counter % self._freq == 0:
            cnt = 0.
            sum_map = defaultdict(float)
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