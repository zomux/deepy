#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging, os
logging.basicConfig(level=logging.INFO)

import numpy as np
import theano
import theano.tensor as T

from deepy.dataset import MnistDataset, MiniBatches
from deepy.networks import ComputationalGraph
from deepy.trainers import optimize_updates
from tutorial2 import MyJointTrainingModel

model_path = os.path.join(os.path.dirname(__file__), "models", "tutorial3.gz")


if __name__ == '__main__':

    model = ComputationalGraph(input_dim=28 * 28)

    parameters = model.parameters
    gradients = T.grad(model.output, parameters)

    gradient_updates, _ = optimize_updates(parameters, gradients,
                                           config={"method": "MOMENTUM",
                                                   "learning_rate": 0.03})

    train_monitors = dict(model.training_monitors)
    test_monitors = dict(model.testing_monitors)

    train_monitors["cost"] = model.output
    test_monitors["cost"] = model.test_output

    train_iteration = theano.function(inputs=model.input_variables,
                                      outputs=train_monitors.values(),
                                      updates=gradient_updates,
                                      allow_input_downcast=True)

    valid_iteration = theano.function(inputs=model.input_variables,
                                     outputs=test_monitors.values(),
                                     allow_input_downcast=True)

    max_epochs = 10

    mnist = MiniBatches(MnistDataset(), batch_size=20)

    for i in range(max_epochs):
        # Training
        cost_matrix = []
        for inputs in mnist.train_set():
            costs = train_iteration(*inputs)
            cost_matrix.append(costs)
        train_costs = list(zip(train_monitors.keys(), np.mean(cost_matrix, axis=0)))
        print "train", i, train_costs

        # Test with valid data
        cost_matrix = []
        for inputs in mnist.valid_set():
            costs = valid_iteration(*inputs)
            cost_matrix.append(costs)
        valid_costs = list(zip(test_monitors.keys(), np.mean(cost_matrix, axis=0)))
        print "valid", i, valid_costs


    model.save_params(model_path)