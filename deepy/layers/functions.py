#!/usr/bin/env python
# -*- coding: utf-8 -*-

from concatenate import Concatenate

def concatenate(vars, axis=1):
    """
    A utility function of concatenate.
    """
    return Concatenate(axis=axis).compute(*vars)