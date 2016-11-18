#!/usr/bin/env python
# -*- coding: utf-8 -*-
from theano.compile import ViewOp
from theano.gradient import DisconnectedType


class DisconnectedGrad(ViewOp):
    def grad(self, args, g_outs):
        return [ DisconnectedType()() for g_out in g_outs]

    def connection_pattern(self, node):
        return [[False]]


disconnected_grad = DisconnectedGrad()