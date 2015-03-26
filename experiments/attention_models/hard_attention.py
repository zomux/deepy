#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os
import logging

import numpy as np

from deepy import NetworkConfig, TrainerConfig, NeuralClassifier, AdaDeltaTrainer
from deepy.networks import NeuralLayer
from deepy.dataset import MnistDataset, MiniBatches
from deepy.util import build_activation
from deepy.util.functions import FLOATX


logging.basicConfig(level=logging.INFO)
import theano
import theano.tensor as T


class HardAttentionLayer(NeuralLayer):

    def __init__(self, activation='relu'):
        super(HardAttentionLayer, self).__init__(10, activation)

    def connect(self, config, vars, x, input_n, id="UNKNOWN"):
        self._config = config
        self._vars = vars
        self.input_n = input_n
        self.id = id
        self.x = x
        self._setup_params()
        self._setup_functions()
        self.connected = True

    def _output_func(self):
        self.x = self.x.reshape((28, 28))
        [l_ts, h_ts, a_ts, wl_grads], _ = theano.scan(fn=self._core_network,
                         outputs_info=[self.l0, self.h0, None, None],
                         non_sequences=[self.x],
                         n_steps=5)

        self.positions = l_ts
        self.last_decision = T.argmax(a_ts[-1])
        wl_grad = T.sum(wl_grads, axis=0) / wl_grads.shape[0]
        self.wl_grad = wl_grad
        return a_ts[-1].reshape((1,10))

    def _setup_functions(self):
        self._assistive_params = []
        self._relu = build_activation("tanh")
        self._tanh = build_activation("tanh")
        self._softmax = build_activation("softmax")
        self.output_func = self._output_func()

    def _setup_params(self):

        self.W_g0 = self.create_weight(7*14, 128, suffix="g0")
        self.W_g1 = self.create_weight(2, 128, suffix="g1")
        self.W_g2_hg = self.create_weight(128, 256, suffix="g2_hg")
        self.W_g2_hl = self.create_weight(128, 256, suffix="g2_hl")

        self.W_h_g = self.create_weight(256, 256, suffix="h_g")
        self.W_h = self.create_weight(256, 256, suffix="h")
        self.B_h = self.create_bias(256, suffix="h")
        self.h0 = self.create_vector(256, "h0")
        self.l0 = self.create_vector(2, "l0")
        self.l0.set_value(np.array([-1, -1], dtype=FLOATX))

        self.W_l = self.create_weight(256, 2, suffix="l")
        self.W_l.set_value(self.W_l.get_value() / 10)
        self.B_l = self.create_bias(2, suffix="l")
        self.W_a = self.create_weight(256, 10, suffix="a")
        self.B_a = self.create_bias(10, suffix="a")


        self.W = [self.W_g0, self.W_g1, self.W_g2_hg, self.W_g2_hl, self.W_h_g, self.W_h, self.W_a]
        self.B = [self.B_h, self.B_a]
        self.params = [self.W_l]

############


mnist = MiniBatches((MnistDataset()), batch_size=20)

model_path = "/tmp/mnist_att_params.gz"

net_conf = NetworkConfig(input_size=28*28)
attention_layer = HardAttentionLayer()
net_conf.layers = [attention_layer]

trainer_conf = TrainerConfig()
trainer_conf.learning_rate = 0.02
trainer_conf.weight_l2 = 0.0001
trainer_conf.hidden_l2 = 0.0001
trainer_conf.monitor_frequency = trainer_conf.validation_frequency = trainer_conf.test_frequency = 1
trainer_conf.test_frequency = 10

network = NeuralClassifier(net_conf)
network.inputs[0].tag.test_value = mnist.valid_set()[0][0]
network.inputs[1].tag.test_value = mnist.valid_set()[0][1]


if os.path.exists(model_path) and True:
    network.load_params(model_path)
    # import pdb;pdb.set_trace()
    # sys.exit()

trainer = AdaDeltaTrainer(network, config=trainer_conf)

start_time = time.time()
c = 1
for k in list(trainer.train(mnist.train_set(), mnist.valid_set(), mnist.test_set())):
    if c > 10:
        break
    c += 1
print k
end_time = time.time()

network.save_params(model_path)

print "time:", ((end_time - start_time )/ 60), "mins"
