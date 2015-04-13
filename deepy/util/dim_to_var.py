#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano.tensor as T

def dim_to_var(ndim, name="k"):
    if ndim == 1:
        return T.vector(name)
    elif ndim == 2:
        return T.matrix(name)
    elif ndim == 3:
        return T.tensor3(name)
    elif ndim == 4:
        return T.tensor4(name)
    else:
        raise NotImplementedError