#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from functions import FLOATX
import os, gzip, cPickle as pickle
import logging as loggers
logging = loggers.getLogger(__name__)

def create_var(theano_tensor, dim=0, test_shape=None, test_dtype=FLOATX):
    """
    Wrap a Theano tensor into the variable for defining neural network.
    :param dim: last dimension of tensor, 0 indicates that the last dimension is flexible
    :rtype: TensorVar
    """
    from deepy.layers.var import NeuralVariable
    var = NeuralVariable(theano_tensor, dim=dim)
    if test_shape:
        if type(test_shape) != list and type(test_shape) != tuple:
            var.set_test_value(test_shape)
        else:
            var.set_test_value(np.random.rand(*test_shape).astype(test_dtype))
    return var

def fill_parameters(path, networks, exclude_free_params=False, check_parameters=False):
        """
        Load parameters from file to fill all network sequentially.
        """
        if not os.path.exists(path):
            raise Exception("model {} does not exist".format(path))
        # Decide which parameters to load
        normal_params = sum([nn.parameters for nn in networks], [])
        all_params = sum([nn.all_parameters for nn in networks], [])
        # Load parameters
        if path.endswith(".gz"):
            opener = gzip.open if path.lower().endswith('.gz') else open
            handle = opener(path, 'rb')
            saved_params = pickle.load(handle)
            handle.close()
            # Write parameters
            if len(all_params) != len(saved_params):
                logging.warning("parameters in the network: {}, parameters in the dumped model: {}".format(len(all_params), len(saved_params)))
            for target, source in zip(all_params, saved_params):
                if not exclude_free_params or target not in normal_params:
                    target.set_value(source)
        elif path.endswith(".npz"):
            arrs = np.load(path)
            # Write parameters
            if len(all_params) != len(arrs.keys()):
                logging.warning("parameters in the network: {}, parameters in the dumped model: {}".format(len(all_params), len(arrs.keys())))
            for target, idx in zip(all_params, range(len(arrs.keys()))):
                if not exclude_free_params or target not in normal_params:
                    source = arrs['arr_%d' % idx]
                    target.set_value(source)
        else:
            raise Exception("File format of %s is not supported, use '.gz' or '.npz' or '.uncompressed.gz'" % path)