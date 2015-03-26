#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

def resource(name):
    """
    Return the path of internal resource.
    """
    dirname = os.path.dirname(__file__)
    deepy_path = os.path.abspath(os.path.join(dirname, "../"))
    return os.path.join(deepy_path, "resources", name)