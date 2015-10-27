#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logging.basicConfig(level=logging.INFO)

from deepy import *
from variational_autoencoder import VariationalAutoEncoder


model_path = os.path.join(os.path.dirname(__file__), "models", "vae1.gz")

def create_model(load=False):
    model = VariationalAutoEncoder(input_dim=28*28)
    model.stack_encoders(Dense(400, 'tanh', init=GaussianInitializer(), random_bias=True))
    model.stack_reparameterization_layer(20)
    model.stack_decoders(Dense(400, 'tanh', init=GaussianInitializer(), random_bias=True),
                         Dense(28*28, 'sigmoid', init=GaussianInitializer()))
    if load:
        model.load_params(model_path)
    return model

if __name__ == '__main__':

    model = create_model()

    trainer = AdamTrainer(model, {"learning_rate": LearningRateAnnealer.learning_rate(0.01)})

    mnist = MiniBatches(MnistDataset(for_autoencoder=True), batch_size=100)

    trainer.run(mnist, controllers=[LearningRateAnnealer(trainer)])

    model.save_params(model_path)