#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from theano import gof, tensor
import theano
import theano.tensor as T


class SampleMultivariateGaussian(gof.Op):
    __props__ = ()

    def make_node(self, x, cov):
        if x.type.ndim != 1:
            raise TypeError('x must be a 1-d vector')
        if cov.type.ndim != 2:
            raise TypeError('cov must be a 2-d matrix')
        return gof.Apply(self, [x, cov], [T.vector(dtype="float32")])

    def perform(self, node, inp, out):
        x, cov = inp
        z, = out
        z[0] = np.random.multivariate_normal(x, cov, 1)[0].astype("float32")

    def grad(self, inputs, outputs):
        return [outputs[0], T.zeros_like(inputs[1])]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)