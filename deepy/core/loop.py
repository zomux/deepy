#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: use smart_replace_graph to make this possible
from deepy.utils import scanner

class Loop(object):

    def __init__(self, sequences=None, outputs_info=None, non_sequences=None, **kwargs):
        """
        A loop function to support "with" grammar.
        """
        if type(sequences) != dict or type(outputs_info) != dict or type(non_sequences) != dict:
            raise Exception("Arguments of Loop shall be dicts.")
        self._sequences = sequences
        self._non_seqeuences = non_sequences
        self._outputs_info = outputs_info
        self._kwargs = kwargs

    def get(self, name):
        pass
