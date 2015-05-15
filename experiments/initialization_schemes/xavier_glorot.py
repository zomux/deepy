#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from util import run
from deepy.utils import XavierGlorotInitializer

model_path = os.path.join(os.path.dirname(__file__), "models", "xavier_glorot1.gz")

if __name__ == '__main__':
    run(XavierGlorotInitializer(), model_path)