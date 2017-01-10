#!/usr/bin/env python
# -*- coding: utf-8 -*-


SENT_MARK = "</s>"
NULL_MARK = "<null>"
UNK_MARK = "<unk>"

import numpy as np
import logging as loggers

logging = loggers.getLogger(__name__)


class Vocab(object):

    def __init__(self, is_lang=True, char_based=False, null_mark=False):
        self.vocab_map = {}
        self.reversed_map = None
        self.size = 0
        self._char_based = char_based
        self.null_mark = null_mark
        if null_mark:
            self.add(NULL_MARK)
        if is_lang:
            self.add(SENT_MARK)
            self.add(UNK_MARK)

    def add(self, word):
        if word not in self.vocab_map:
            self.vocab_map[word] = self.size
            self.size += 1

    def index(self, word):
        if word in self.vocab_map:
            return self.vocab_map[word]
        else:
            return self.vocab_map[UNK_MARK]

    def word(self, index):
        if not self.reversed_map:
            self.reversed_map = {}
            for k in self.vocab_map:
                self.reversed_map[self.vocab_map[k]] = k
        return self.reversed_map[index]

    def transform(self, word):
        v = np.zeros(self.size, dtype=int)
        v[self.index(word)] = 1
        return v

    def transform_index(self, index):
        v = np.zeros(self.size, dtype=int)
        v[index] = 1
        return v

    def _load_fixed_size(self, path, max_size):
        from collections import Counter
        logging.info("fixed size: %d" % max_size)
        counter = Counter()
        for line in open(path).readlines():
            line = line.strip()
            words = line.split(" ") if not self._char_based else line
            counter.update(words)
        for w, _ in counter.most_common(max_size):
            self.add(w)

    def load(self, path, max_size=-1):
        logging.info("load data from %s" % path)
        if max_size > 0:
            self._load_fixed_size(path, max_size)
            return
        for line in open(path).xreadlines():
            line = line.strip()
            words = line.split(" ") if not self._char_based else line
            map(self.add, words)
        logging.info("vocab size: %d" % self.size)

    @property
    def sent_index(self):
        return self.index(SENT_MARK)

    @property
    def sent_vector(self):
        return self.transform(SENT_MARK)


