#!/usr/bin/env python
# -*- coding: utf-8 -*-

from func import run_experiment, ERROR_KEYWORD


def test():
    result = run_experiment("mnist/deep_convolution.py", timeout=30)
    assert ERROR_KEYWORD not in result
