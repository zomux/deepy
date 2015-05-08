#!/usr/bin/env python
# -*- coding: utf-8 -*-

from  random import Random
random = Random(3)
from deepy.networks import NeuralRegressor
from deepy.layers import Dense
from deepy.trainers import SGDTrainer
from deepy.conf import TrainerConfig
import threading

GAMMA = 0.9
EPSILON = 0.2
EXPERIENCE_SIZE = 5000
EXPERIENCE_RECORD_INTERVAL = 10
REPLAY_TIMES = 20
TDERROR_CLAMP = 1.0
LEARNING_RATE = 0.01
HIDDEN_UNITS = 100

def get_model(state_num, action_num):
    model = NeuralRegressor(state_num)
    model.stack(Dense(HIDDEN_UNITS, activation='tanh'),
                Dense(action_num))
    return model

class DQNAgent(object):
    """
    Agent of deep Q learning.
    """

    def __init__(self, state_num, action_num, experience_replay=True):
        self.state_num = state_num
        self.action_num = action_num
        self.experience_replay = experience_replay
        self.experience_pool = []
        self.model = get_model(state_num, action_num)
        train_conf = TrainerConfig()
        train_conf.learning_rate = LEARNING_RATE
        self.trainer = SGDTrainer(self.model, train_conf)
        self.trainer.training_names = []
        self.trainer.training_variables = []
        self.thread_lock = threading.Lock()
        self.epsilon = EPSILON
        self.tick = 0

    def action(self, state):
        if random.uniform(0, 1) < EPSILON:
            return random.randint(0, self.action_num -1)
        else:
            action = self.model.compute([state])
            return int(action[0].argmax())

    def learn(self, state, action, reward, next_state, enable_replay=True):
        # Learn
        next_q = self.model.compute([next_state])[0]
        best_a = next_q.argmax()
        max_q = next_q[best_a]
        target = reward + GAMMA * max_q
        # Forward
        with self.thread_lock:
            y = list(self.model.compute([state])[0])
        y_action = y[action]
        half_clamp = TDERROR_CLAMP / 2
        if target > y_action + half_clamp:
            target = y_action + half_clamp
        elif target < y_action - half_clamp:
            target = y_action - half_clamp
        y[action] = target
        # Back-propagate
        with self.thread_lock:
         self.trainer.learning_func([state], [y])
        # Replay
        if self.experience_replay and enable_replay:
            if self.tick % EXPERIENCE_RECORD_INTERVAL == 0:
                self.record_experience(state, action, reward, next_state)
            self.tick += 1
            self.replay()

    def replay(self):
        if not self.experience_pool:
            return
        for _ in range(REPLAY_TIMES):
            state, action, reward, next_state = random.choice(self.experience_pool)
            self.learn(state, action, reward, next_state, False)

    def record_experience(self, state, action, reward, next_state):
        if len(self.experience_pool) >= EXPERIENCE_SIZE:
            self.experience_pool.pop(0)
        self.experience_pool.append((state, action, reward, next_state))

    def save(self, path):
        self.model.save_params(path)

    def load(self, path):
        self.model.load_params(path)

    def set_epsilon(self, value):
        self.epsilon = value


