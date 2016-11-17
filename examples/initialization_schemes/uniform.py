#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from util import run
from deepy.utils import UniformInitializer

model_path = os.path.join(os.path.dirname(__file__), "models", "uniform1.gz")

if __name__ == '__main__':
    run(UniformInitializer(), model_path)