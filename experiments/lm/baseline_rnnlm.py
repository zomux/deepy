#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from argparse import ArgumentParser

from utils import load_data
from lm import NeuralLM
from deepy.trainers import SGDTrainer, LearningRateAnnealer, AdamTrainer
from deepy.layers import LSTM, Dense, RNN, Softmax3D
from layers import FullOutputLayer


logging.basicConfig(level=logging.INFO)

default_model = os.path.join(os.path.dirname(__file__), "models", "baseline_rnnlm.gz")

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--small", action="store_true")
    args = ap.parse_args()

    vocab, lmdata = load_data(small=args.small, history_len=5, batch_size=64)
    model = NeuralLM(vocab.size)
    model.stack(RNN(hidden_size=100, output_type="sequence", hidden_activation="sigmoid",
                    persistent_state=True, batch_size=lmdata.size,
                    reset_state_for_input=0),
                FullOutputLayer(vocab.size))

    if os.path.exists(args.model):
        model.load_params(args.model)

    trainer = SGDTrainer(model, {"learning_rate": LearningRateAnnealer.learning_rate(1.2),
                                 "weight_l2": 1e-7})
    annealer = LearningRateAnnealer(trainer)

    trainer.run(lmdata, controllers=[annealer])

    model.save_params(default_model)
