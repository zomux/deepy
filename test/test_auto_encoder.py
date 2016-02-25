#!/usr/bin/env python
# -*- coding: utf-8 -*-

from func import run_experiment, ERROR_KEYWORD


def test():
    result = run_experiment("highway_networks/mnist_highway.py", timeout=60)
    assert ERROR_KEYWORD not in result
