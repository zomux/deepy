#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest

from deepy.utils.functions import VarMap


class FunctionsTest(unittest.TestCase):

    def test_varmap(self):
        vars = VarMap()
        vars.x = 1
        print vars.x
        print vars.y