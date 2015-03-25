#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from nlpy.basic import DefaultLemmatizer
import sys, os

if __name__ == '__main__':
    lem = DefaultLemmatizer()
    for l in sys.stdin.xreadlines():
        l = l.strip()
        ws = l.split(" ")
        print " ".join(map(lem.lemmatize, ws))