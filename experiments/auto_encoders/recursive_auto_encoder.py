#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging, os
logging.basicConfig(level=logging.INFO)

from deepy.networks import RecursiveAutoEncoder
from deepy.trainers import SGDTrainer, LearningRateAnnealer

from util import get_data, VECTOR_SIZE

model_path = os.path.join(os.path.dirname(__file__), "models", "rae1.gz")

if __name__ == '__main__':
    model = RecursiveAutoEncoder(input_dim=VECTOR_SIZE, rep_dim=VECTOR_SIZE)

    trainer = SGDTrainer(model)

    annealer = LearningRateAnnealer(trainer)

    trainer.run(get_data(), controllers=[annealer])

    model.save_params(model_path)

