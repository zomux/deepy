#!/usr/bin/env python
# -*- coding: utf-8 -*-

class TrainingController(object):
    """
    Abstract class of training controllers.
    """


    def __init__(self, trainer):
        """
        :type trainer: deepy.trainers.base.NeuralTrainer
        """
        self._trainer = trainer

    def invoke(self):
        """
        Return True to exit training.
        """
        return False