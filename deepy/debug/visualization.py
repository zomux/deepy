#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

chars = [u" ", u"▁", u"▂", u"▃", u"▄", u"▅", u"▆", u"▇", u"█"]


def plot_hinton(arr, max_arr=None):
    if max_arr == None: max_arr = arr
    arr = np.array(arr)
    max_val = max(abs(np.max(max_arr)), abs(np.min(max_arr)))
    print (np.array2string(arr,
                          formatter={'float_kind': lambda x: visual_hinton(x, max_val)},
                          max_line_width=5000
    ))


def visual_hinton(val, max_val):
    if abs(val) == max_val:
        step = len(chars) - 1
    else:
        step = int(abs(float(val) / max_val) * len(chars))
    colourstart = ""
    colourend = ""
    if val < 0: colourstart, colourend = '\033[90m', '\033[0m'
    #bh.internal = colourstart + chars[step] + colourend
    return colourstart + chars[step] + colourend