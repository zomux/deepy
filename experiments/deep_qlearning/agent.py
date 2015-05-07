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
    model.stack_layers(Dense(100, activation='tanh'),
                       Dense(action_num))
    return model

GAMMA = 0.9
EPSILON = 0.2

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
        if random.uniform(0, 1) < EPSILON:
            return random.randint(0, self.action_num -1)
        else:
            action = self.model.compute([state])
            return int(action[0].argmax())

    def learn(self, state, action, reward, next_state):
        next_q = self.model.compute([next_state])[0]
        best_a = next_q.argmax()
        max_q = next_q[best_a]
        y = list(self.model.compute([state])[0])
        y[action] = reward + GAMMA * max_q
        self.trainer.learning_func([state], [y])

    def save(self, path):
        self.model.save_params(path)

    def load(self, path):
        self.model.load_params(path)


