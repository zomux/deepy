#!/usr/bin/env python
# -*- coding: utf-8 -*-

from  random import Random
random = Random(3)
from deepy.networks import NeuralRegressor
from deepy.layers import Dense, Softmax
from deepy.trainers import SGDTrainer
from deepy.conf import TrainerConfig

def get_model(state_num, action_num):
    model = NeuralRegressor(state_num)
    model.stack_layers(
                       Dense(30, activation='tanh'),
                       Dense(30, activation='tanh'),
                       Dense(action_num),
                       Softmax())
    return model

GAMMA = 0.9

class RLAgent(object):
    """
    Agent of deep Q learning.
    """

    def __init__(self, state_num, action_num):
        self.state_num = state_num
        self.action_num = action_num
        self.model = get_model(state_num, action_num)
        train_conf = TrainerConfig()
        train_conf.learning_rate = 0.01
        self.trainer = SGDTrainer(self.model, train_conf)
        self.trainer.training_names = []
        self.trainer.training_variables = []

    def action(self, state):
        action = self.model.compute([state])
        return int(action[0].argmax())

    def learn(self, state, action, reward, next_state):
        next_q = self.model.compute([next_state])[0]
        best_a = next_q.argmax()
        max_q = next_q[best_a]
        y = list(self.model.compute([state])[0])
        y[action] = reward + GAMMA * max_q
        self.trainer.learning_func([state], [y])


