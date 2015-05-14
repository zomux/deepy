#!/usr/bin/env python
# -*- coding: utf-8 -*-

class LineIterator(object):

    def __init__(self, path):
        self._path = path

    def __iter__(self):
        return (line.strip() for line in open(self._path).xreadlines())