#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from util import run

model_path = os.path.join(os.path.dirname(__file__), "models", "mlp_cg1.gz")

if __name__ == '__main__':
    run("cg", model_path)