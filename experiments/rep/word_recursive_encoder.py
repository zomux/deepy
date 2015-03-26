#!/usr/bin/env python
# -*- coding: utf-8 -*-


#!/usr/bin/env python
# -*- coding: utf-8 -*-



import time
import os
import random as rnd
import logging

import numpy as np
import theano
import theano.tensor as T

from deepy import NetworkConfig, TrainerConfig, AdaGradTrainer
from deepy.util.functions import FLOATX
from deepy.networks import NeuralLayer
from deepy.networks.recursive import GeneralAutoEncoder
from nlpy.util import LineIterator, FakeGenerator
from deepy.util import build_activation
from deepy.trainers.minibatch_optimizer import MiniBatchOptimizer


logging.basicConfig(level=logging.INFO)

random = rnd.Random(3)

"""
Word recursive encoder layer.
This is a simple recursive auto-encoder perform gradual encoding by merge two nodes together.
"""
class WRELayer(NeuralLayer):

    def __init__(self, size, activation='tanh', noise=0., dropouts=0., beta=0.,
                 optimization="ADAGRAD", unfolding=True, additional_h=False, max_reg=4, deep=False, batch_size=10,
                 realtime_update=True):
        """
        Recursive autoencoder layer follows the path of a given parse tree.
        Manually accumulate gradients.
        """
        super(WRELayer, self).__init__(size, activation, noise, dropouts)
        self.size = size
        self.learning_rate = 0.01
        self.disable_bias = True
        self.optimization = optimization
        self.beta = beta
        self.unfolding = unfolding
        self.max_reg = max_reg
        self.deep = deep
        self.batch_size = batch_size
        self.realtime_update = realtime_update
        self.encode_optimizer = MiniBatchOptimizer(batch_size=self.batch_size, realtime=realtime_update)
        self.decode_optimizer = MiniBatchOptimizer(batch_size=self.batch_size, realtime=realtime_update)

    def connect(self, config, vars, x, input_n, id="UNKNOWN"):
        """
        Connect to a network
        :type config: deepy.conf.NetworkConfig
        :type vars: deepy.functions.VarMap
        :return:
        """
        self._config = config
        self._vars = vars
        self.input_n = input_n
        self.id = id
        self.x = x
        self._setup_params()
        self._setup_functions()
        self.connected = True

    def updating_callback(self):
        if not self.realtime_update:
            self.encode_optimizer.run()
            self.decode_optimizer.run()

    def _preprocess_step(self, d):
        rep = self._activation_func(T.dot(d, self.W_in) + self.B_in)
        return rep

    def _postprocess_step(self, rep):
        out = self._activation_func(T.dot(rep, self.W_out) + self.B_out)
        out = self._softmax_func(out)
        return out

    def _recursive_func(self):

        step_num = self.x.shape[0] - 1

        # Pre-processing
        input_reps = self._preprocess_step(self.x)

        # Encoding
        internal_reps, _ = theano.scan(self._recursive_encode_step,
                                  sequences=[T.arange(input_reps.shape[0], 1, -1)],
                                  outputs_info=[input_reps],
                                  non_sequences=[])

        top_reps = internal_reps[-1][:1]
        # Unfolding
        top_reps = T.set_subtensor(input_reps[:1], top_reps)
        decoded_reps, _ = theano.scan(self._recursive_decode_step,
                                  sequences=[T.arange(1, input_reps.shape[0])],
                                  outputs_info=[top_reps],
                                  non_sequences=[])

        output_rep = decoded_reps[-1]
        # Post-processing
        outpus = self._postprocess_step(output_rep)

        # Cost function
        cost = T.sum(-T.log(outpus[:, T.argmax(self.x, axis=1)]))


        return top_reps[-1], cost


    def _layer_encode_step(self, i, reps):
        output_rep = self._activation_func(T.dot(reps[i], self.W_e1) + T.dot(reps[i+1], self.W_e2) + self.B_e)

        return output_rep

    def _recursive_encode_step(self, input_len, input_reps):
        output_reps, _ = theano.scan(self._layer_encode_step,
                                  sequences=[T.arange(0, input_len - 1)],
                                  non_sequences=[input_reps])

        return T.set_subtensor(input_reps[:output_reps.shape[0]], output_reps)

    def _layer_deocde_step(self, i, reps):
        output_rep = self._activation_func(T.dot(reps[i], self.W_d1) + T.dot(reps[i+1], self.W_d2) + self.B_d)
        return output_rep

    def _recursive_decode_step(self, input_len, input_reps):
        real_input_reps = T.concatenate([self.zero_rep, input_reps[:input_len], self.zero_rep])
        output_reps, _ = theano.scan(self._layer_deocde_step,
                                     sequences=[T.arange(0, real_input_reps.shape[0] - 1)],
                                     non_sequences=[real_input_reps])

        return T.set_subtensor(input_reps[:output_reps.shape[0]], output_reps)



    def encode_func(self):
        return T.sum(self._vars.p)
        # seq_len = self._vars.seq.shape[0]
        # # Encoding
        # [reps, _], _ = theano.scan(self._encode_step, sequences=[T.arange(seq_len)],
        #                                    outputs_info=[None, self.init_registers],
        #                                    non_sequences=[self.x, self._vars.seq])
        #
        # return reps



    def decode_func(self):
        # Not implemented
        return T.sum(self._vars.p)

    def _setup_functions(self):
        self._assistive_params = []
        self._activation_func = build_activation(self.activation)
        self._softmax_func = build_activation('softmax')
        top_rep, self.output_func = self._recursive_func()
        # self.predict_func, self.predict_updates = self._encode_func()
        self.monitors.append(("top_rep<0.1", 100 * (abs(top_rep) < 0.1).mean()))
        self.monitors.append(("top_rep<0.9", 100 * (abs(top_rep) < 0.9).mean()))
        self.monitors.append(("top_rep:mean", abs(top_rep).mean()))

    def _setup_params(self):

        self.W_e1 = self.create_weight(self.size, self.size, "enc1")
        self.W_e2 = self.create_weight(self.size, self.size, "enc2")
        self.B_e = self.create_bias(self.size, "enc")

        self.W_d1 = self.create_weight(self.size, self.size, "dec1")
        self.W_d2 = self.create_weight(self.size, self.size, "dec2")
        self.B_d = self.create_bias(self.size, "dec")

        self.W_in = self.create_weight(self.input_n, self.size, "in")
        self.W_out = self.create_weight(self.size, self.input_n, "out")

        self.B_in = self.create_bias(self.size, "in")
        self.B_out = self.create_bias(self.input_n, "out")

        self.zero_rep = self.create_matrix(1, self.size, "zero")


        self.W = [self.W_e1, self.W_e2, self.W_d1, self.W_d2, self.W_in, self.W_out]
        self.B = [self.B_e, self.B_d, self.B_in, self.B_out]
        self.params = []

        # Just for decoding
        self._vars.p = T.vector("p", dtype=FLOATX)


"""
WRE data loader
"""

ENGLISH_CHAT_SET = "abcdefghijklmnopqrstuvwxyz~!@#$%^()_+[]{}/\"':;<>,.-="

class WordDataBuilder(object):

    def __init__(self, path, chat_set=None, build_valid_set=True):
        self.chat_set = chat_set if chat_set else ENGLISH_CHAT_SET
        self.input_size = len(self.chat_set) + 1
        data = self._build_data(path)
        random.shuffle(data)
        if build_valid_set:
            valid_size = int(len(data) * 0.2)
            self._train_data = data[valid_size:]
            self._valid_data = data[:valid_size]
        else:
            self._train_data = data
            self._valid_data = []

    def _build_data(self, path):
        data = []
        for l in LineIterator(path):
            l = l.lower()
            chars = filter(lambda x: x in self.chat_set, l)
            if not chars:
                continue
            char_ids = [self.chat_set.index(c) + 1 for c in chars]
            char_ids = [0] + char_ids + [0]
            word_data = [np.eye(1,M=self.input_size,k=c)[0] for c in char_ids]
            data.append([word_data])
        return data

    def get_valid_data(self):
        random.shuffle(self._valid_data)
        return self._valid_data.__iter__()

    def get_train_data(self):
        random.shuffle(self._valid_data)
        return self._train_data.__iter__()

    def train_data(self):
        return FakeGenerator(self, "get_train_data")

    def valid_data(self):
        return FakeGenerator(self, "get_valid_data")

def get_wre_network(model_path=""):
    net_conf = NetworkConfig(input_size=300)
    net_conf.layers = [WRELayer(size=300)]

    trainer_conf = TrainerConfig()
    trainer_conf.learning_rate = 0.01
    trainer_conf.weight_l2 = 0.0001
    trainer_conf.hidden_l2 = 0.0001
    trainer_conf.monitor_frequency = trainer_conf.validation_frequency = trainer_conf.test_frequency = 1

    network = GeneralAutoEncoder(net_conf)
    if os.path.exists(model_path):
        network.load_params(model_path)
    return network

if __name__ == '__main__':

    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("premodel")
    ap.add_argument("model")
    args = ap.parse_args()

    print "[ARGS]", args

    builder = WordDataBuilder("/home/hadoop/data/morpho/data/topwords.1500.txt")

    """
    Setup network
    """
    pretrain_model = args.premodel
    model_path = args.model

    net_conf = NetworkConfig(input_size=builder.input_size)
    net_conf.layers = [WRELayer(size=100)]

    trainer_conf = TrainerConfig()
    trainer_conf.learning_rate = 0.01
    trainer_conf.weight_l2 = 0.0001
    trainer_conf.hidden_l2 = 0.0001
    trainer_conf.monitor_frequency = trainer_conf.validation_frequency = trainer_conf.test_frequency = 1

    network = GeneralAutoEncoder(net_conf)

    trainer = AdaGradTrainer(network, config=trainer_conf)

    """
    Run the network
    """
    start_time = time.time()

    if os.path.exists(pretrain_model):
        network.load_params(pretrain_model)
    # elif os.path.exists(model_path):
    #     network.load_params(model_path)

    c = 0
    for _ in trainer.train(builder.train_data(), valid_set=builder.valid_data()):
        c += 1
        if c > 20:
           pass
        pass

    end_time = time.time()
    network.save_params(model_path)

    print "elapsed time:", (end_time - start_time) / 60, "mins"

