#!/usr/bin/env python
# -*- coding: utf-8 -*-

from visualization import plot_hinton
from detect_nan import DETECT_NAN_MODE
from monitor import monitor_var_sum, monitor_var

try:
    from ipdb import set_trace
except ImportError as e:
    from pdb import set_trace