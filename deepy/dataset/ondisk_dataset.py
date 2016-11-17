#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import types
from . import Dataset
from data_processor import DataProcessor
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
                 cached=False, post_processing=None, shuffle_memory=False, data_processor=None):
        """
        :type data_processor: DataProcessor
        """
        self._train_path = train_path
        self._valid_path = valid_path
        self._test_path = test_path
        self._train_size = train_size
        self._cache_on_memory = cached
        self._cached_train_data = None
        self._post_processing = post_processing if post_processing else lambda x: x
        self._shuffle_memory = shuffle_memory
        self._epoch = 0
        self._data_processor = data_processor
        if data_processor and not isinstance(data_processor, DataProcessor):
            raise Exception("data_processor must be an instance of DataProcessor.")
        if self._cache_on_memory:
            logging.info("Cache on memory")
            self._cached_train_data = list(map(self._post_processing, StreamPickler.load(open(self._train_path))))
            self._train_size = len(self._cached_train_data)
            if self._shuffle_memory:
                logging.info("Shuffle on-memory data")
                global_rand.shuffle(self._cached_train_data)

    def _process_data(self, split, epoch, dataset):
        if self._data_processor:
            return self._data_processor.process(split, epoch, dataset)
        else:
            return dataset

    def generate_train_data(self):
        self._epoch += 1
        data_source = self._cached_train_data if self._cache_on_memory else StreamPickler.load(open(self._train_path))
        for data in self._process_data('train', self._epoch, data_source):
            yield self._post_processing(data)

    def generate_valid_data(self):
        data_source = StreamPickler.load(open(self._valid_path))
        for data in self._process_data('valid', self._epoch, data_source):
            yield self._post_processing(data)

    def generate_test_data(self):
        data_source = StreamPickler.load(open(self._test_path))
        for data in self._process_data('test', self._epoch, data_source):
            yield self._post_processing(data)

    def train_set(self):
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
