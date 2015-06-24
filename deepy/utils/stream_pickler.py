#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
  from cPickle import dumps, loads
except ImportError:
  from pickle import dumps, loads

class StreamPickler(object):
    """
    Pickle massive objects without using too much memory.
    The code of this class is modified upon https://code.google.com/p/streaming-pickle/.
    """

    @staticmethod
    def dump(iterable_to_pickle, file_obj):
        """
        dump contents of an iterable iterable_to_pickle to file_obj, a file
        opened in write mode
        """
        for elt in iterable_to_pickle:
            StreamPickler.dump_one(elt, file_obj)

    @staticmethod
    def dump_one(elt_to_pickle, file_obj):
        """
        dumps one element to file_obj, a file opened in write mode
        """
        pickled_elt_str = dumps(elt_to_pickle)
        file_obj.write(pickled_elt_str)
        # record separator is a blank line
        # (since pickled_elt_str might contain its own newlines)
        file_obj.write('\n\n')

    @staticmethod
    def load(file_obj):
        """
        load contents from file_obj, returning a generator that yields one
        element at a time
        """
        cur_elt = []
        for line in file_obj:
          cur_elt.append(line)

          if line == '\n':
            pickled_elt_str = ''.join(cur_elt)
            cur_elt = []
            try:
                elt = loads(pickled_elt_str)
            except ValueError:
                continue

            yield elt
