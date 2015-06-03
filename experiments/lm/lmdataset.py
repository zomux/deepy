#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging as loggers
import numpy as np
from deepy.dataset import Dataset
from deepy.utils import FLOATX

logging = loggers.getLogger(__name__)

class LMDataset(Dataset):


    def __init__(self, vocab, train_path, valid_path, history_len=-1, char_based=False, max_tokens=999, min_tokens=0, sort=True):
        """
        Generate data for training with RNN
        :type vocab: vocab.Vocab
        """
        assert history_len == -1 or history_len >= 1
        self.vocab = vocab
        self.history_len = history_len
        self.char_based = char_based
        self.sort = sort

        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

        self._train_set = self.read_data(train_path)
        self._valid_set = self.read_data(valid_path)

    def train_set(self):
        return self._train_set

    def valid_set(self):
        return self._valid_set

    def read_data(self, path):
        data = []
        sent_count = 0
        for line in open(path).xreadlines():
            line = line.strip()
            wc = len(line) if self.char_based else line.count(" ") + 1
            if wc < self.min_tokens or wc > self.max_tokens:
                continue
            sent_count += 1
            sequence = [self.vocab.sent_index]
            tokens = line if self.char_based else line.split(" ")
            for w in tokens:
                sequence.append(self.vocab.index(w))
            sequence.append(self.vocab.sent_index)

            if self.history_len == -1:
                # Full sentence
                data.append(self.convert_to_data(sequence))
            else:
                # trunk by trunk
                for begin in range(0, len(sequence), self.history_len):
                    trunk = sequence[begin: begin + self.history_len + 1]
                    if len(trunk) > 1:
                        data.append(self.convert_to_data(trunk))
        if self.sort:
            data.sort(key=lambda x: len(x[1]))
        logging.info("loaded from %s: %d sentences, %d data pieces " % (path, sent_count, len(data)))
        return data

    def convert_to_data(self, seq):
        assert len(seq) >= 2
        input_indices = seq[:-1]
        target_indices = seq[1:]
        return input_indices, target_indices


