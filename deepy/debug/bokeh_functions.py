#!/usr/bin/env python
# -*- coding: utf-8 -*-


def bokeh_start(session_id, url, dryrun=False):
    try:
        from bokeh_logger import BokehLogger
    except ImportError as e:
        print (e.message)
        raise Exception("Please install bokeh.")
    global bokeh_logger
    if "bokeh_logger" not in globals() or not bokeh_logger:
        if dryrun:
            bokeh_logger = "dryrun"
        else:
            bokeh_logger = BokehLogger(session_id, url)


def bokeh_record(name, value):
    global bokeh_logger
    if "bokeh_logger" not in globals() or not bokeh_logger:
        raise Exception("Please call bokeh_start first.")
    if bokeh_logger == "dryrun":
        return
    bokeh_logger.record(name, value)


def bokeh_save(prefix):
    global bokeh_logger
    if "bokeh_logger" not in globals() or not bokeh_logger:
        raise Exception("Please call bokeh_start first.")
    if bokeh_logger == "dryrun":
        return
    bokeh_logger.save(prefix)
