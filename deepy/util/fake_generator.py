#!/usr/bin/env python
# -*- coding: utf-8 -*-


class FakeGenerator(object):

  def __init__(self, dataset, method_name):
    self.dataset = dataset
    self.method_name = method_name

  def __iter__(self):
    return getattr(self.dataset, self.method_name)()