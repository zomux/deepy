#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging

from vocab import Vocab
from lmdataset import LMDataset
from deepy.dataset import BunchSequences


logging.basicConfig(level=logging.INFO)

resource_dir = os.path.abspath(os.path.dirname(__file__)) + os.sep + "resources"

def load_data(small=True, char_based=False, batch_size=20, vocab_size=10000, history_len=5, max_tokens=50):
    vocab_path = os.path.join(resource_dir, "ptb.train.txt")
    valid_path = os.path.join(resource_dir, "ptb.valid.txt")
    if small:
        train_path = os.path.join(resource_dir, "ptb.train.10k.txt")
    else:
        train_path = os.path.join(resource_dir, "ptb.train.txt")
    vocab = Vocab(char_based=char_based)
    vocab.load(vocab_path, max_size=vocab_size)

    lmdata = LMDataset(vocab, train_path, valid_path, history_len=-1, char_based=char_based, max_tokens=max_tokens)
    batch = BunchSequences(lmdata, batch_size=batch_size, fragment_length=history_len)
    return vocab, batch
