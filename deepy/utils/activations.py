#!/usr/bin/env python
# -*- coding: utf-8 -*-


import functools

import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from deepy.utils.functions import FLOATX


theano_rng = RandomStreams(seed=3)


def add_noise(x, sigma, rho):
    if sigma > 0 and rho > 0:
        noise = theano_rng.normal(size=x.shape, std=sigma, dtype=FLOATX)
        mask = theano_rng.binomial(size=x.shape, n=1, p=1-rho, dtype=FLOATX)
        return mask * (x + noise)
    if sigma > 0:
        return x + theano_rng.normal(size=x.shape, std=sigma, dtype=FLOATX)
    if rho > 0:
        mask = theano_rng.binomial(size=x.shape, n=1, p=1-rho, dtype=FLOATX)
        return mask * x
    return x

def softmax(x):
    # T.nnet.softmax doesn't work with the HF trainer.
    z = T.exp(x.T - x.T.max(axis=0))
    return (z / z.sum(axis=0)).T

def build_activation(act=None):
        def compose(a, b):
            c = lambda z: b(a(z))
            c.__theanets_name__ = '%s(%s)' % (b.__theanets_name__, a.__theanets_name__)
            return c
        if '+' in act:
            return functools.reduce(
                compose, (build_activation(a) for a in act.split('+')))
        options = {
            'tanh': T.tanh,
            'linear': lambda z: z,
            'logistic': T.nnet.sigmoid,
            'sigmoid': T.nnet.sigmoid,
            'softplus': T.nnet.softplus,
            'softmax': softmax,
            'theano_softmax': T.nnet.softmax,

            # shorthands
            'relu': lambda z: z * (z > 0),
            'trel': lambda z: z * (z > 0) * (z < 1),
            'trec': lambda z: z * (z > 1),
            'tlin': lambda z: z * (abs(z) > 1),

            # modifiers
            'rect:max': lambda z: T.minimum(1, z),
            'rect:min': lambda z: T.maximum(0, z),

            # normalization
            'norm:dc': lambda z: (z.T - z.mean(axis=1)).T,
            'norm:max': lambda z: (z.T / T.maximum(1e-10, abs(z).max(axis=1))).T,
            'norm:std': lambda z: (z.T / T.maximum(1e-10, T.std(z, axis=1))).T,
            }
        for k, v in options.items():
            v.__theanets_name__ = k
        try:
            return options[act]
        except KeyError:
            raise KeyError('unknown activation %r' % act)