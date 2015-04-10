import time
import os
from collections import Counter
import random as rnd

from experiments.lm import Vocab
from deepy import NetworkConfig, TrainerConfig
from deepy.trainers import MomentumTrainer
from deepy.util.functions import FLOATX
from deepy.layers.simple_rnn import SimpleRNN, SimpleRNNLayer
from deepy.util import LineIterator, FakeGenerator


random = rnd.Random(3)
import theano.tensor as T

import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

######

class ConversationDataGenerator(object):

    def __init__(self, vocab_size = 100):
        self.vocab = Vocab()
        self.topic_vocab = Vocab(is_lang=False)
        self.raw_data = []
        self.max_state = 0
        self.vocab_size = vocab_size

    def addFile(self, path, topic="<unk>", is_noise=False):
        state = 1
        self.topic_vocab.add(topic)
        for l in LineIterator(path):
            if not l: state = 1
            if is_noise:
                state = 0
            if not l.startswith("a :"):
                continue
            sent = l.split(":")[1].strip()
            self.raw_data.append((sent, topic, state))
            if state > self.max_state:
                self.max_state = state
            state += 1
        self._buildVocab()

    def buildData(self):
        self.data = []
        for sent, t, y in self.raw_data:
            words = sent.split(' ')
            x = map(self.vocab.binvector, words)
            t = np.array(self.topic_vocab.binvector(t), dtype=FLOATX)
            self.data.append((x, [y], t))

    def fetch(self, shuffle=True):
        indexes = range(len(self.data))
        if shuffle:
            random.shuffle(indexes)
        for i in indexes:
            yield self.data[i]

    def getGenerator(self):
        return FakeGenerator(self, "fetch")

    def _buildVocab(self):
        counter = Counter()
        self.vocab = Vocab()
        for sent, t, _ in self.raw_data:
            words = sent.split(' ')
            counter.update(words)
        for w, _ in counter.most_common(self.vocab_size):
            self.vocab.add(w)


class CustomizedClassificationLayer(SimpleRNNLayer):

    def __init__(self, size, topic_size, target_size=-1):
        super(CustomizedClassificationLayer, self).__init__(size, target_size)
        self.topic_size = topic_size

    def _recurrent_step(self, x_t, h_t):
        h = self._activation_func(T.dot(x_t, self.W_i)+ T.dot(h_t, self.W_r))

        s = self._softmax_func(T.dot(h, self.W_s) + T.dot(self._vars.t, self.W_t))
        return [h ,s]

    def _extra_params(self):
        self._vars.t = T.vector("t", dtype=FLOATX)
        self.W_t, _, _ = self.create_params(self.topic_size, self.target_size, "topic")
        self.W.append(self.W_t)
        self.inputs.append(self._vars.t)

#######


model_path = "/tmp/rnn_conversation2.gz"
data_path = "/home/hadoop/data/conversations/tok"

dg = ConversationDataGenerator()
fn_list = sorted(os.listdir(data_path))
for fn in fn_list:
    if "travel" not in fn:
        continue
    topic = fn.split(".")[0]
    dg.addFile(data_path + os.sep + fn, topic=topic)

# Negative sampling
for fn in fn_list:
    if "travel" not in fn:
        continue
    topic = fn.split(".")[0]
    outdomain_data = [x[0] for x in dg.raw_data if x[1] != topic]
    random.shuffle(outdomain_data)
    for d in outdomain_data[:20]:
        dg.raw_data.append((d, topic, 0))

valid_dg = ConversationDataGenerator()
valid_dg.vocab = dg.vocab
valid_dg.topic_vocab = dg.topic_vocab
random.shuffle(dg.raw_data)
valid_size = len(dg.raw_data) / 10
valid_dg.raw_data = dg.raw_data[:valid_size]
dg.raw_data = dg.raw_data[valid_size:]
dg.buildData()
valid_dg.buildData()
data = dg.getGenerator()
valid_data = valid_dg.getGenerator()

net_conf = NetworkConfig(input_size=dg.vocab.size)
net_conf.layers = [CustomizedClassificationLayer(size=50, topic_size=dg.topic_vocab.size, target_size=dg.max_state + 1)]

trainer_conf = TrainerConfig()
trainer_conf.learning_rate = 0.01
trainer_conf.weight_l2 = 0.0001
trainer_conf.hidden_l2 = 0.0001
trainer_conf.monitor_frequency = trainer_conf.validation_frequency = trainer_conf.test_frequency = 1

network = SimpleRNN(net_conf)

trainer = MomentumTrainer(network, config=trainer_conf)

start_time = time.time()

if os.path.exists(model_path):
    network.load_params(model_path)
    pass

def test(sent):
    global dg, network
    words = sent.split()
    x = map(dg.vocab.binvector, words)
    t = dg.topic_vocab.binvector("travel13")
    return network.classify(x, t)[-1]

def train_error():
    global dg, network, data
    errors = []
    for x, y, t in data:
        err = 1 if network.classify(x, t)[-1] != y[-1] else 0
        errors.append(err)
    return np.mean(np.array(errors))

# print test("is there a restroom we can use in this area ?")

import pdb; pdb.set_trace()
raise SystemExit


for _ in trainer.train(data, valid_set=valid_data):
    pass

end_time = time.time()
network.save_params(model_path)


print "elapsed time:", (end_time - start_time) / 60, "mins"
