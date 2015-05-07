#!/usr/bin/env python
# -*- coding: utf-8 -*-

from  random import Random
random = Random(3)


class RLAgent(object):
    """
    Agent of deep Q learning.
    """

    def __init__(self, state_num, action_num):
        self.state_num = state_num
        self.action_num = action_num

    def act(self, state):
        return random.randint(0, self.action_num)

    def learn(self, state, action, reward, next_state):
        pass