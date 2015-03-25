#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import unittest

from deepy.functions import VarMap


class FunctionsTest(unittest.TestCase):

    def test_varmap(self):
        vars = VarMap()
        vars.x = 1
        print vars.x
        print vars.y