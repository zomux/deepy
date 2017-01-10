#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging, os
logging.basicConfig(level=logging.INFO)

from deepy.networks import RecursiveAutoEncoder
from deepy.trainers import SGDTrainer, LearningRateAnnealer

from util import get_data, VECTOR_SIZE

model_path = os.path.join(os.path.dirname(__file__), "models", "rae1.gz")

if __name__ == '__main__':
    model = RecursiveAutoEncoder(input_dim=VECTOR_SIZE, rep_dim=10)

    trainer = SGDTrainer(model)

    annealer = LearningRateAnnealer()

    trainer.run(get_data(), epoch_controllers=[annealer])

    model.save_params(model_path)

