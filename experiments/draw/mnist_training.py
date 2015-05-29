#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging, os
logging.basicConfig(level=logging.INFO)

from deepy.trainers import AdamTrainer, LearningRateAnnealer
from deepy.dataset import BinarizedMnistDataset, MiniBatches
from deepy.conf import TrainerConfig

from core import DrawModel

model_path = os.path.join(os.path.dirname(__file__), "models", "mnist1.gz")

if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("--load", default="", help="pre-trained model path")
    args = ap.parse_args()

    model = DrawModel(image_width=28, image_height=28, attention_times=64)

    if args.load:
        model.load_params(args.load)

    conf = TrainerConfig()
    conf.gradient_clipping = 10
    conf.learning_rate = LearningRateAnnealer.learning_rate(0.004)
    conf.weight_l2 = 0
    #conf.avoid_nan = True
    # from deepy import DETECT_NAN_MODE
    # conf.theano_mode = DETECT_NAN_MODE
    # TODO: Find out the problem causing NaN
    trainer = AdamTrainer(model, conf)

    mnist = MiniBatches(BinarizedMnistDataset(), batch_size=100)

    trainer.run(mnist, controllers=[LearningRateAnnealer(trainer)])

    model.save_params(model_path)
