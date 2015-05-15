#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from util import run
from deepy.utils import GaussianInitializer

model_path = os.path.join(os.path.dirname(__file__), "models", "gaussian1.gz")

if __name__ == '__main__':
    # I have to set std to be 0.1 in this case, or it will not convergence
    run(GaussianInitializer(deviation=0.1), model_path)