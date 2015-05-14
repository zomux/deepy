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
from deepy.utils.functions import FLOATX
from deepy.networks import NeuralLayer
from deepy.layers.recursive import GeneralAutoEncoder
from nlpy.util import LineIterator, FakeGenerator
from deepy.utils import build_activation


logging.basicConfig(level=logging.INFO)

random = rnd.Random(3)

"""
Word recursive encoder layer.
This is a simple recursive auto-encoder perform gradual encoding by merge two nodes together.
"""
class RecurrentEncoderLayer(NeuralLayer):

    def __init__(self, size, activation='tanh', noise=0., dropouts=0.,
                 deep=False, deep2=False, batch_size=10, reverse_input=True, partial_training=0):
        """
        Recurrent auto encoder.
        """
        super(RecurrentEncoderLayer, self).__init__(size, activation, noise, dropouts)
        self.size = size
        self.deep = deep
        self.deep2 = deep2
        self.partial_training = partial_training
        self.batch_size = batch_size
        self.reverse_input = reverse_input


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

    def _recursive_func(self):
        # Encoding
        internal_reps, _ = theano.scan(self._encode_step,
                                       sequences=[self._preprocess(self.x)],
                                       outputs_info=[self.h0])
        # Deep encoding
        if self.deep:
            internal_reps, _ = theano.scan(self._deep_encode_step,
                                           sequences=[internal_reps],
                                           outputs_info=[self.h0])
        last_rep = internal_reps[-1]
        # Decoding
        decoded_reps, _ = theano.scan(self._decode_step,
                                           outputs_info=[last_rep],
                                           n_steps=self.x.shape[0])
        if self.deep:
            decoded_reps, _ = theano.scan(self._deep_decode_step,
                                           sequences=[decoded_reps],
                                           outputs_info=[self.h0])
        # Output
        outputs = self._activation_func(T.dot(decoded_reps, self.W_do) + self.B_do)

        # Cost function
        # cost = T.mean(-T.log(outputs[:, T.argmax(self.x, axis=1)]))
        cost = T.mean(T.sum((outputs - self.x)**2, axis=1))

        err = T.mean(T.neq(T.argmax(self.x, axis=1), T.argmax(outputs, axis=1)))
        self.monitors.append(("err", err))

        return last_rep, cost

    def _deep_encode_step(self, input, h):
        preact = T.dot(input, self.W_ei2) + T.dot(h, self.W_eh2)
        next_h = self._activation_func(preact)
        return next_h

    def _deep_encode_step2(self, input, h):
        preact = T.dot(input, self.W_ei3) + T.dot(h, self.W_eh3)
        next_h = self._activation_func(preact)
        return next_h

    def _encode_step(self, input, h):
        preact = T.dot(input, self.W_ei) + T.dot(h, self.W_eh) + self.B_eh
        next_h = self._activation_func(preact)
        return next_h

    def _deep_decode_step(self, input, h):
        preact = T.dot(input, self.W_di2) + T.dot(h, self.W_dh2)
        next_h = self._activation_func(preact)
        return next_h

    def _deep_decode_step2(self, input, h):
        preact = T.dot(input, self.W_di3) + T.dot(h, self.W_dh3)
        next_h = self._activation_func(preact)
        return next_h

    def _decode_step(self, h):
        next_h = self._activation_func(T.dot(h, self.W_dh) + self.B_dh)
        return next_h


    def encode_func(self):
        enc_input, = self.encode_inputs
        internal_reps, _ = theano.scan(self._encode_step,
                                       sequences=[self._preprocess(enc_input)],
                                       outputs_info=[self.h0])
        # Deep encoding
        if self.deep:
            internal_reps, _ = theano.scan(self._deep_encode_step,
                                           sequences=[internal_reps],
                                           outputs_info=[self.h0])
        last_rep = internal_reps[-1]
        return last_rep

    def decode_func(self):
        dec_input, = self.decode_inputs
        [_, outputs], _ = theano.scan(self._decode_step,
                                      outputs_info=[dec_input, None],
                                      n_steps=25)
        return outputs

    def _preprocess(self, input):
        if self.reverse_input:
            return input[::-1][1:]
        else:
            return input[:-1]


    def _setup_functions(self):
        self._activation_func = build_activation(self.activation)
        self._softmax_func = build_activation('softmax')
        top_rep, self.output_func = self._recursive_func()
        self.monitors.append(("top_rep<0.1", 100 * (abs(top_rep) < 0.1).mean()))
        self.monitors.append(("top_rep<0.9", 100 * (abs(top_rep) < 0.9).mean()))
        self.monitors.append(("top_rep:mean", abs(top_rep).mean()))

    def _setup_params(self):

        self.W_ei = self.create_weight(self.input_n, self.size, "ei")
        self.W_eh = self.create_weight(self.size, self.size, "h")
        self.B_eh = self.create_bias(self.size, "h")
        self.h0 = self.create_vector(self.size, "h0")

        self.W_dh = self.create_weight(self.size, self.size, "dh")
        self.W_do = self.create_weight(self.size, self.input_n, "do")
        self.B_dh = self.create_bias(self.size, "dh")
        self.B_do = self.create_bias(self.input_n, "do")

        self.W = [self.W_ei, self.W_eh, self.W_dh, self.W_do]
        self.B = [self.B_eh, self.B_dh, self.B_do]


        if self.deep:
            self.W_ei2 = self.create_weight(self.size, self.size, "ei2")
            self.W_eh2 = self.create_weight(self.size, self.size, "eh2")
            self.W_dh2 = self.create_weight(self.size, self.size, "dh2")
            self.W_di2 = self.create_weight(self.size, self.size, "di2")
            self.W.extend([self.W_ei2, self.W_eh2, self.W_dh2, self.W_di2])

        if self.deep2:
            self.W_ei3 = self.create_weight(self.size, self.size, "ei3")
            self.W_eh3 = self.create_weight(self.size, self.size, "eh3")
            self.W_dh3 = self.create_weight(self.size, self.size, "dh3")
            self.W_di3 = self.create_weight(self.size, self.size, "di3")

            if self.partial_training == 2:
                self.parameters = self.W + self.B
                self.W = [self.W_ei3, self.W_eh3, self.W_dh3, self.W_di3]
                self.B = []
            else:
                self.W.extend([self.W_ei2, self.W_eh2, self.W_dh2, self.W_di2])

        self.encode_inputs = [T.matrix("encode_input")]
        self.decode_inputs = [T.vector("decode_input")]


"""
WRE data loader
"""

ENGLISH_CHAT_SET = "abcdefghijklmnopqrstuvwxyz~!@#$%^()_+[]{}/\"':;<>,.-="

class CharVocab(object):

    def __init__(self, chat_set=None):
        self.chat_set = chat_set if chat_set else ENGLISH_CHAT_SET
        self.size = len(self.chat_set) + 1

    def convert(self, word):
        chars = filter(lambda x: x in self.chat_set, word)
        if not chars:
            return np.array([])
        char_ids = [self.chat_set.index(c) + 1 for c in chars]
        char_ids = char_ids + [0]
        word_data = [np.eye(1, M=self.size, k=c)[0] for c in char_ids]
        word_data = np.array(word_data, dtype=FLOATX)
        return word_data

    def restore(self, data):
        char_ids = np.argmax(data, axis=1)
        chars = []
        for char_id in char_ids:
            if char_id == 0:
                break
            else:
                chars.append(self.chat_set[char_id - 1])
        return "".join(chars)

class WordDataBuilder(object):

    def __init__(self, path, chat_set=None, build_valid_set=True):
        self.vocab = CharVocab(chat_set)
        self.input_size = self.vocab.size
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
            word_data = self.vocab.convert(l)
            if len(word_data) == 0:
                continue
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

def get_network(model_path=""):
    net_conf = NetworkConfig(input_size=100)
    net_conf.layers = [RecurrentEncoderLayer(size=300)]

    trainer_conf = TrainerConfig()
    trainer_conf.learning_rate = 0.01
    trainer_conf.weight_l2 = 0.0001
    trainer_conf.hidden_l2 = 0.0001
    trainer_conf.monitor_frequency = trainer_conf.validation_frequency = trainer_conf.test_frequency = 1

    network = GeneralAutoEncoder(net_conf)
    if os.path.exists(model_path):
        network.load_params(model_path)
    return network

default_char_vocab = CharVocab()

def encode_word(network, word):
    global default_char_vocab
    d = default_char_vocab.convert(word)
    rep = network.encode(d)
    return rep

def decode_word(network, d):
    global default_char_vocab
    output = network.decode(d)
    word = default_char_vocab.restore(output)
    return word

if __name__ == '__main__':

    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("premodel")
    ap.add_argument("model")
    args = ap.parse_args()

    print "[ARGS]", args

    builder = WordDataBuilder(args.data)

    """
    Setup network
    """
    pretrain_model = args.premodel
    model_path = args.model

    net_conf = NetworkConfig(input_size=builder.input_size)
    net_conf.layers = [RecurrentEncoderLayer(size=100, deep=True)]

    trainer_conf = TrainerConfig()
    trainer_conf.learning_rate = 0.01
    trainer_conf.weight_l2 = 0.0001
    trainer_conf.hidden_l2 = 0.0001
    trainer_conf.patience = 50
    trainer_conf.monitor_frequency = trainer_conf.validation_frequency = trainer_conf.test_frequency = 1

    network = GeneralAutoEncoder(net_conf)

    if os.path.exists(pretrain_model):
        network.load_params(pretrain_model)
        # import pdb;pdb.set_trace()

    # trainer = AdaDeltaTrainer(network, trainer_conf)
    # trainer = ScipyTrainer(network, 'cg')
    trainer = AdaGradTrainer(network, trainer_conf)

    """
    Run the network
    """
    start_time = time.time()


    for _ in trainer.train(builder.train_data(), valid_set=builder.valid_data()):
        pass

    end_time = time.time()
    network.save_params(model_path)

    print "elapsed time:", (end_time - start_time) / 60, "mins"

