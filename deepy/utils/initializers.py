#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out

class WeightInitializer(object):
    """
    Initializer for creating weights.
    """

    def __init__(self, seed=None):
        if not seed:
            seed = 3
        self.rand = np.random.RandomState(seed)

    def sample(self, shape):
        """
        Sample parameters with given shape.
        """
        raise NotImplementedError

class UniformInitializer(WeightInitializer):
    """
    Uniform weight sampler.
    """

    def __init__(self, scale=None, svd=False, seed=None):
        super(UniformInitializer, self).__init__(seed)
        self.scale = scale
        self.svd = svd

    def sample(self, shape):
        if not self.scale:
            scale = np.sqrt(6. / sum(get_fans(shape)))
        else:
            scale = self.scale
        weight = self.rand.uniform(-1, 1, size=shape) * scale
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

    def __init__(self, mean=0, deviation=0.01, seed=None):
        super(GaussianInitializer, self).__init__(seed)
        self.mean = mean
        self.deviation = deviation

    def sample(self, shape):
        weight = self.rand.normal(self.mean, self.deviation, size=shape)
        return weight

class IdentityInitializer(WeightInitializer):
    """
    Initialize weight as identity matrices.
    """

    def __init__(self, scale=1):
        super(IdentityInitializer, self).__init__()
        self.scale = 1

    def sample(self, shape):
        assert len(shape) == 2
        return np.eye(*shape) * self.scale

class XavierGlorotInitializer(WeightInitializer):
    """
    Xavier Glorot's weight initializer.
    See http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    """

    def __init__(self, uniform=False, seed=None):
        """
        Parameters:
            uniform - uniform distribution, default Gaussian
            seed - random seed
        """
        super(XavierGlorotInitializer, self).__init__(seed)
        self.uniform = uniform

    def sample(self, shape):
        scale = np.sqrt(2. / sum(get_fans(shape)))
        if self.uniform:
            return self.rand.uniform(-1, 1, size=shape) * scale
        else:
            return self.rand.randn(*shape) * scale

class KaimingHeInitializer(WeightInitializer):
    """
    Kaiming He's initialization scheme, especially made for ReLU.
    See http://arxiv.org/abs/1502.01852.
    """
    def __init__(self, uniform=False, seed=None):
        """
        Parameters:
            uniform - uniform distribution, default Gaussian
            seed - random seed
        """
        super(KaimingHeInitializer, self).__init__(seed)
        self.uniform = uniform

    def sample(self, shape):
        fan_in, fan_out = get_fans(shape)
        scale = np.sqrt(2. / fan_in)
        if self.uniform:
            return self.rand.uniform(-1, 1, size=shape) * scale
        else:
            return self.rand.randn(*shape) * scale

class OrthogonalInitializer(WeightInitializer):
    """
    Orthogonal weight initializer.
    """
    def __init__(self, scale=1.1, seed=None):
        """
        Parameters:
            scale - scale
            seed - random seed
        """
        super(OrthogonalInitializer, self).__init__(seed)
        self.scale = scale

    def sample(self, shape):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return self.scale * q[:shape[0], :shape[1]]