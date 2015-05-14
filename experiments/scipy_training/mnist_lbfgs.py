#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from util import run

"""
Some errors happened on my machine, so L-BFGS optimizer in scipy fails.
- Raphael Shu, 2015/5
"""

model_path = os.path.join(os.path.dirname(__file__), "models", "mlp_lbfgs1.gz")

if __name__ == '__main__':
    run("l-bfgs-b", model_path)