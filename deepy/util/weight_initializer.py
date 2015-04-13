#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from . import FLOATX

rand = np.random.RandomState(3)

class WeightInitializer(object):
    """
    Initializer for creating weights.
    """

    def sample(self, shape):
        """
        Sample parameters with given shape.
        """
        raise NotImplementedError

class UniformInitializer(WeightInitializer):
    """
    Uniform weight sampler.
    """

    def __init__(self, scale=None, svd=False):
        self.scale = scale
        self.svd = svd

    def sample(self, shape):
        if not self.scale:
            scale = np.sqrt(6. / sum(shape))
        else:
            scale = self.scale
        weight = rand.uniform(-1, 1, size=shape) * scale
        if self.svd:
            norm = np.sqrt((weight**2).sum())
            ws = scale * weight / norm
            _, v, _ = np.linalg.svd(ws)
            ws = scale * ws / v[0]
        return weight

class GaussianInitializer(WeightInitializer):
    """
    Gaussian weight sampler.
    """

    def __init__(self, mean=0, deviation=0.001):
        self.mean = mean
        self.deviation = deviation

    def sample(self, shape):
        weight = rand.normal(self.mean, self.deviation, size=shape)
        return weight

class IdentityInitializer(WeightInitializer):
    """
    Initialize weight as identity matrices.
    """

    def __init__(self, scale=1):
        self.scale = 1

    def sample(self, shape):
        assert len(shape) == 2
        return np.eye(*shape) * self.scale