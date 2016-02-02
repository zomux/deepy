#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging, os
logging.basicConfig(level=logging.INFO)
import theano.tensor as T

from deepy.dataset import MnistDataset, MiniBatches
from deepy.networks import ComputationalGraph
from deepy.layers import Dense, Softmax, NeuralLayer, Chain
from deepy.trainers import MomentumTrainer, LearningRateAnnealer
from deepy.utils import AutoEncoderCost, CrossEntropyCost, ErrorRateCost


model_path = os.path.join(os.path.dirname(__file__), "models", "tutorial2.gz")

class MyJointTrainingModel(NeuralLayer):
    """
    A customized model that trains an auto-encoder and MLP classifier simultaneously.
    """

    def __init__(self, internal_layer_size=100):
        super(MyJointTrainingModel, self).__init__("my joint-training model")
        self.internal_layer_size = internal_layer_size

    def prepare(self):
        """
        All codes that create parameters should be put into 'setup' function.
        """
        self.output_dim = 10
        self.encoder = Chain(self.input_dim).stack(Dense(self.internal_layer_size, 'tanh'))
        self.decoder = Chain(self.internal_layer_size).stack(Dense(self.input_dim))
        self.classifier = Chain(self.internal_layer_size).stack(Dense(50, 'tanh'),
                                                      Dense(self.output_dim),
                                                      Softmax())

        self.register_inner_layers(self.encoder, self.decoder, self.classifier)

        self.target_input = T.ivector('target')
        self.register_external_inputs(self.target_input)

    def compute_tensor(self, x):
        """
        Build the computation graph here.
        """
        internal_variable = self.encoder.compute_tensor(x)

        decoding_output = self.decoder.compute_tensor(internal_variable)

        classification_output = self.classifier.compute_tensor(internal_variable)

        auto_encoder_cost = AutoEncoderCost(decoding_output, x).get()

        classification_cost = CrossEntropyCost(classification_output, self.target_input).get()

        final_cost = 0.01 * auto_encoder_cost + classification_cost

        error_rate = ErrorRateCost(classification_output, self.target_input).get()

        self.register_monitors(("err", error_rate),
                               ("encoder_cost", auto_encoder_cost),
                               ("classify_cost", classification_cost))

        return final_cost


if __name__ == '__main__':
    model = ComputationalGraph(input_dim=28 * 28)

    mnist = MiniBatches(MnistDataset(), batch_size=20)

    trainer = MomentumTrainer(model, {'weight_l2': 0.0001})

    trainer.run(mnist, controllers=[LearningRateAnnealer(trainer)])

    model.save_params(model_path)