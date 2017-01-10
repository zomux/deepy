#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import logging as loggers
import os
import re

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams as SharedRandomStreams

logging = loggers.getLogger(__name__)
logging.setLevel(loggers.INFO)

"""
This file is deprecated.
"""

FLOATX = theano.config.floatX
EPSILON = T.constant(1.0e-15, dtype=FLOATX)
BIG_EPSILON = T.constant(1.0e-7, dtype=FLOATX)

if 'DEEPY_SEED' in os.environ:
    global_seed = int(os.environ['DEEPY_SEED'])
    logging.info("set global random seed to %d" % global_seed)
else:
    global_seed = 3
global_rand = np.random.RandomState(seed=global_seed)
global_theano_rand = RandomStreams(seed=global_seed)
global_shared_rand = SharedRandomStreams(seed=global_seed)


def make_float_matrices(*names):
    ret = []
    for n in names:
        ret.append(T.matrix(n, dtype=FLOATX))
    return ret


def make_float_vectors(*names):
    ret = []
    for n in names:
        ret.append(T.vector(n, dtype=FLOATX))
    return ret


def back_grad(jacob, err_g):
    return T.dot(jacob, err_g)
    # return (jacob.T * err_g).T

def build_node_name(n):
    if "owner" not in dir(n) or "inputs" not in dir(n.owner):
        return str(n)
    else:
        op_name = str(n.owner.op)
        if "{" not in op_name:
            op_name = "Elemwise{%s}" % op_name
        if "," in op_name:
            op_name = re.sub(r"\{([^}]+),[^}]+\}", "{\\1}", op_name)
        if "_" in op_name:
            op_name = re.sub(r"\{[^}]+_([^_}]+)\}", "{\\1}", op_name)
        return "%s(%s)" % (op_name, ",".join([build_node_name(m) for m in n.owner.inputs]))


class VarMap():
    def __init__(self):
        self.varmap = {}

    def __get__(self, instance, owner):
        if instance not in self.varmap:
            return None
        else:
            return self.varmap[instance]

    def __set__(self, instance, value):
        self.varmap[instance] = value

    def __contains__(self, item):
        return item in self.varmap

    def update_if_not_existing(self, name, value):
        if name not in self.varmap:
            self.varmap[name] = value

    def get(self, name):
        return self.varmap[name]

    def set(self, name, value):
        self.varmap[name] = value