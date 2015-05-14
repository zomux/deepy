#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import numpy as np

def detect_nan(i, node, fn):
    for output in fn.outputs:
        if (not isinstance(output[0], np.random.RandomState) and
            np.isnan(output[0]).any()):
            print '*** NaN detected ***'
            theano.printing.debugprint(node)
            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            import pdb;pdb.set_trace()

DETECT_NAN_MODE = theano.compile.MonitorMode(post_func=detect_nan)