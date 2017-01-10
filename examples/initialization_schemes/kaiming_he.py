#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from util import run
from deepy.utils import KaimingHeInitializer

model_path = os.path.join(os.path.dirname(__file__), "models", "kaiming_he1.gz")

if __name__ == '__main__':
    run(KaimingHeInitializer(), model_path)