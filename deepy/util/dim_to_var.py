#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano.tensor as T

def dim_to_var(ndim):
    if ndim == 1:
        return T.vector()
    elif ndim == 2:
        return T.matrix()
    elif ndim == 3:
        return T.tensor3()
    elif ndim == 4:
        return T.tensor4()
    else:
        raise NotImplementedError