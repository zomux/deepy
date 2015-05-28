#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Detect the source which produces NaN.

Usage
---

conf.theano_mode = DETECT_NAN_MODE

Note
---

Be sure to use theano flag 'optimizer=None'.
"""

import theano
import numpy as np

def detect_nan(i, node, fn):
    if str(node.op).startswith("GPU_mrg_uniform"):
        return
    for output in fn.outputs:
        if (not isinstance(output[0], np.random.RandomState) and
            np.isnan(output[0]).any()):
            print '*** NaN detected ***'
            theano.printing.debugprint(node)
            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            import pdb;pdb.set_trace()

DETECT_NAN_MODE = theano.compile.MonitorMode(post_func=detect_nan)
