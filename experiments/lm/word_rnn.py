#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from argparse import ArgumentParser

from utils import load_data
from lm import NeuralLM
from deepy.trainers import SGDTrainer, LearningRateAnnealer
from deepy.layers import LSTM, Dense, RNN


logging.basicConfig(level=logging.INFO)

default_model = os.path.join(os.path.dirname(__file__), "models", "word_rnn1.gz")

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("--model", default=default_model)
    ap.add_argument("--small", action="store_ture")
    args = ap.parse_args()

    vocab, lmdata = load_data(small=args.small, history_len=-1)
    #import pdb;pdb.set_trace()
    model = NeuralLM(input_dim=vocab.size, input_tensor=3)
    model.stack(RNN(hidden_size=100, output_type="sequence", hidden_activation='tanh'),
                Dense(vocab.size, activation="softmax"))

    if os.path.exists(args.model):
        model.load_params(args.model)

    trainer = SGDTrainer(model, {"learning_rate": LearningRateAnnealer.learning_rate(0.1)})
    annealer = LearningRateAnnealer(trainer)

    trainer.run(lmdata, controllers=[annealer])

    model.save_params(args.model)
