#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from bokeh.client import push_session
from bokeh.plotting import figure, curdoc, reset_output
from bokeh.models import Range1d
from bokeh.io import save


class BokehLogger(object):
    def __init__(self, session_id, url):
        self._session = push_session(curdoc(), session_id=session_id, url=url)
        self._objects = {}
    
    def record(self, name, value):
        if name not in self._objects:
            fig = figure(plot_width=1000, plot_height=200, title=name)
            x = np.array([])
            y = np.array([])
            line = fig.line(x, y, line_width=1)
            scatter = fig.scatter(x, y, size=5)
            self._objects[name] = {"figure": fig, "line": line, "scatter": scatter, "count": 0}
            self._session.document.add_root(fig)
        reset_output()
        obj = self._objects[name]
        obj["count"] += 1
        obj["line"].data_source.stream({"x": [obj["count"]], "y": [value]})
        obj["scatter"].data_source.stream({"x": [obj["count"]], "y": [value]})
        obj["figure"].x_range = Range1d(0, obj["count"] + 100)
        if obj["count"] >= 5:
            y_data = obj["scatter"].data_source.data["y"]
            variance = 2 * np.sqrt(np.var(y_data[-5:]))
            mean = np.mean(y_data[-5:])
            min_val = min([np.min(y_data[-5:]), mean - variance])
            max_val = max([np.max(y_data[-5:]), mean + variance])
            obj["figure"].y_range = Range1d(
                min_val - abs(min_val) * 0.1, max_val + abs(max_val) * 0.1
            )
    
    def save(self, prefix):
        for name, obj in self._objects.items():
            save(obj["figure"], filename="{}_{}.png".format(prefix, name))
