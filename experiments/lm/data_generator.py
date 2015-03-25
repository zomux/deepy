#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from nlpy.util import LineIterator
import sys
import numpy as np
import math
from progressbar import ProgressBar
import random

import logging as loggers

logging = loggers.getLogger(__name__)

class RNNDataGenerator(object):


    def __init__(self, vocab, data_path, history_len, batch_size = 1, overlap=False, progress=False, fixed_length=True,
                 target_vector=False, _just_test=False, shuffle=True, max_words=999, min_words=0):
        """
        Generate data for training with RNN
        :type vocab: nlpy.lm.Vocab
        :type data_path: str
        :param history_len: if this value is -1, then one trunk is a sentence
        :type history_len: int
        :type binvector: bool
        """
        self._vocab = vocab
        self._target_vector = target_vector
        self._just_test = _just_test
        self.history_len = history_len
        self.batch_size = batch_size
        self.minibatch_mode = not (batch_size == 1)
        self.fixed_length = fixed_length
        self.progress = progress
        self.overlap = overlap
        self.shuffle = shuffle

        self.sentences = []

        # Treat each sentence as a trunk
        for line in LineIterator(data_path):
            sequence = [vocab.sent_index]
            wc = line.count(" ") + 1
            if wc < min_words or wc > max_words:
                continue
            for w in line.split(" "):
                sequence.append(vocab.index(w))
            sequence.append(vocab.sent_index)
            self.sentences.append(sequence)
        logging.info("%d sentences loaded from %s" % (len(self.sentences), data_path))

    def sequential_data(self):

        if self.progress:
            progress = ProgressBar(maxval=len(self.sentences)).start()

        trunk_size = self.history_len + 1

        index_sequence = range(len(self.sentences))
        if self.shuffle:
            random.shuffle(index_sequence)

        for j in xrange(len(index_sequence)):
            sent_i = index_sequence[j]
            sent = self.sentences[sent_i]

            if self.history_len == -1:
                trunk_n = 1
                trunk_size = len(sent)
            else:
                trunk_n = int(math.ceil(float(len(sent) - 1) / trunk_size))

            if self.overlap and trunk_size > 0:
                trunk_n = len(sent) - self.history_len - 1

            for trunk_i in range(trunk_n):
                if not self.overlap:
                    start_of_x = trunk_i * trunk_size
                    end_of_x = trunk_i * trunk_size + trunk_size
                else:
                    start_of_x = trunk_i
                    end_of_x = trunk_i + self.history_len + 1

                if end_of_x >= len(sent):
                    end_of_x = len(sent) - 1
                x_indexs = sent[start_of_x: end_of_x]
                y_indexs = sent[start_of_x + 1: end_of_x + 1]
                if len(x_indexs) < trunk_size and self.fixed_length:
                    x_indexs += [0] * (trunk_size - len(x_indexs))
                    y_indexs += [0] * (trunk_size - len(y_indexs))
                assert len(x_indexs) == len(y_indexs)
                x = map(self._vocab.binvector_of_index, x_indexs)
                if self._target_vector:
                    y = map(self._vocab.binvector_of_index, y_indexs)
                else:
                    y = y_indexs
                yield (x, y)

                if self._just_test and j > 100:
                    break

            if self.progress:
                progress.update(j)

        if self.progress:
            # progress.finish()
            progress.update(0)
            sys.stdout.write("\033[F")

    def __iter__(self):


        if not self.minibatch_mode:
            return self.sequential_data()
        else:
            raise NotImplementedError



