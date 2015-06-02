#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging, os
logging.basicConfig(level=logging.INFO)

from deepy.trainers import AdamTrainer, LearningRateAnnealer, FineTuningAdaGradTrainer
from deepy.dataset import BinarizedMnistDataset, MiniBatches

from core import DrawModel

model_path = os.path.join(os.path.dirname(__file__), "models", "mnist1.gz")

if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("--load", default="", help="pre-trained model path")
    ap.add_argument("--finetune", action="store_true")
    args = ap.parse_args()

    model = DrawModel(image_width=28, image_height=28, attention_times=64)

    if args.load:
        model.load_params(args.load)

    conf = {
        "gradient_clipping": 10,
        "learning_rate": LearningRateAnnealer.learning_rate(0.004),
        "weight_l2": 0
    }
    # conf.avoid_nan = True
    # from deepy import DETECT_NAN_MODE
    # conf.theano_mode = DETECT_NAN_MODE
    # TODO: Find out the problem causing NaN
    if args.finetune:
        trainer = FineTuningAdaGradTrainer(model, conf)
    else:
        trainer = AdamTrainer(model, conf)

    mnist = MiniBatches(BinarizedMnistDataset(), batch_size=100)

    trainer.run(mnist, controllers=[])

    model.save_params(model_path)
