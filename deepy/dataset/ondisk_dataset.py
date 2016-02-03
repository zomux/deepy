#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import Dataset
from deepy.utils import FakeGenerator, StreamPickler, global_rand

import logging as loggers
logging = loggers.getLogger(__name__)

class OnDiskDataset(Dataset):
    """
    Load large on-disk dataset.
    The data should be dumped with deepy.utils.StreamPickler.
    You must convert the data to mini-batches before dump it to a file.
    """

    def __init__(self, train_path, valid_path=None, test_path=None, train_size=None,
                 cached=False, post_processing=None, shuffle_memory=False, curriculum=None):
        self._train_path = train_path
        self._valid_path = valid_path
        self._test_path = test_path
        self._train_size = train_size
        self._cache_on_memory = cached
        self._cached_train_data = None
        self._post_processing = post_processing if post_processing else lambda x: x
        self._shuffle_memory = shuffle_memory
        self._curriculum = curriculum
        self._curriculum_count = 0
        if curriculum and not callable(curriculum):
            raise Exception("curriculum function must be callable")
        if curriculum and not cached:
            raise Exception("curriculum learning needs training data to be cached")
        if self._cache_on_memory:
            logging.info("Cache on memory")
            self._cached_train_data = list(map(self._post_processing, StreamPickler.load(open(self._train_path))))
            self._train_size = len(self._cached_train_data)
            if self._shuffle_memory:
                logging.info("Shuffle on-memory data")
                global_rand.shuffle(self._cached_train_data)

    def curriculum_train_data(self):
        self._curriculum_count += 1
        logging.info("curriculum learning: round {}".format(self._curriculum_count))
        return self._curriculum(self._cached_train_data, self._curriculum_count)

    def generate_train_data(self):
        for data in StreamPickler.load(open(self._train_path)):
            yield self._post_processing(data)

    def generate_valid_data(self):
        for data in StreamPickler.load(open(self._valid_path)):
            yield self._post_processing(data)

    def generate_test_data(self):
        for data in StreamPickler.load(open(self._test_path)):
            yield self._post_processing(data)

    def train_set(self):
        if self._cache_on_memory:
            if self._curriculum:
                return FakeGenerator(self, "curriculum_train_data")
            else:
                return self._cached_train_data
        if not self._train_path:
            return None
        return FakeGenerator(self, "generate_train_data")

    def valid_set(self):
        if not self._valid_path:
            return None
        return FakeGenerator(self, "generate_valid_data")

    def test_set(self):
        if not self._test_path:
            return None
        return FakeGenerator(self, "generate_test_data")

    def train_size(self):
        return self._train_size
