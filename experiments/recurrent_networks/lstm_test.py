import os
import sys
from argparse import ArgumentParser
import logging

import numpy as np

from experiments.lm import Vocab
from experiments.lm.data_generator import RNNDataGenerator
from deepy import NetworkConfig, TrainerConfig
from deepy.layers.lstm import RecurrentNetwork, LSTMLayer
from deepy.trainers import SGDTrainer
from deepy.util import Timer, resource


logging.basicConfig(level=logging.INFO)

####

def entropy(network, data):
    logp_sum = 0.0
    wc = 0
    for d in data:
        xs, ys = d
        preds = network.get_probs(xs, ys)
        logps = np.log2(preds)
        logp_sum += sum(logps)
        wc += len(logps)
        if ys[-1] == 0:
            network.clear_hidden()
        # sys.stdout.write("~PPL: %f\r" %  2 ** (- logp_sum / wc))
        sys.stdout.write("~ENTROPY: %f\r" %  (- logp_sum / wc))
    sys.stdout.write("")
    ent = - logp_sum / wc
    return ent


#######

ap = ArgumentParser()
ap.add_argument("premodel", default="/home/hadoop/play/model_zoo/lstm_lm_pretrain4.gz")
ap.add_argument("model", default="/home/hadoop/play/model_zoo/lstm_lm_pretrain4.gz")
ap.add_argument("min", default=0)
ap.add_argument("max", default=999)
args = ap.parse_args()

print "args:", args


pre_model_path = args.premodel
model_path = args.model

train_path = resource("ptb_lm_test/ptb.train.txt")
train_100_path = resource("ptb_lm_test/ptb.train.10k.txt")
valid_path = resource("ptb_lm_test/ptb.valid.txt")
# question_train = "/home/hadoop/data/questions/train.txt"
# question_valid = "/home/hadoop/data/questions/valid.txt"
vocab = Vocab()
vocab.load(train_path)

train_data = RNNDataGenerator(vocab, train_100_path, target_vector=False, overlap=False,
                              history_len=9, _just_test=False, fixed_length=False, progress=True,
                              min_words=int(args.min), max_words=int(args.max))
valid_data = RNNDataGenerator(vocab, valid_path, target_vector=False, overlap=False,
                              history_len=9, _just_test=False, fixed_length=False, progress=False)

net_conf = NetworkConfig(input_size=vocab.size)
net_conf.layers = [LSTMLayer(size=15)]

trainer_conf = TrainerConfig()
trainer_conf.learning_rate = 0.1
trainer_conf.monitor_frequency = trainer_conf.validation_frequency = trainer_conf.test_frequency = 1

network = RecurrentNetwork(net_conf)

# perplexity(network, valid_data)
# raise SystemExit

trainer = SGDTrainer(network, config=trainer_conf)


best_entropy = 9999999.9
if os.path.exists(pre_model_path):
    network.load_params(pre_model_path)
    # best_entropy = entropy(network, valid_data)

def drytest(sent):
    words = sent.split()
    x = map(vocab.binvector, words)
    y = network.classify(x)
    network.clear_hidden()
    return map(vocab.word, y)

def expand(sent):
    words = sent.split()
    while words[-1] != "</s>":
        # x = map(vocab.binvector, words); y = network.classify(x); yw = vocab.word(y[-1]); words.append(yw); words
        x = map(vocab.binvector, words)
        y = network.classify(x)
        yw = vocab.word(y[-1])
        words.append(yw)
        network.clear_hidden()
    return words



"""
Strategy:
RESET -> 3 failures -> NON RESET
3 failures -> RELOAD
9 failures -> END
"""
network.do_reset_grads = True
threshold = 3
failed_count = 0
reload_count = 0

timer = Timer()

c = 0
for _ in trainer.train(train_data):
    print ""
    ent = entropy(network, valid_data)
    if ent < best_entropy:
        best_entropy = ent
        failed_count = 0
        reload_count = 0
        network.save_params(model_path)
    else:
        failed_count += 1
        reload_count += 1
        if network.do_reset_grads and failed_count >= threshold:
            print "Turn off resetting gradient velocities"
            network.load_params(model_path)
            network.do_reset_grads = False
            failed_count = 0
            reload_count = 0
        elif reload_count >= threshold:
            network.load_params(model_path)
            reload_count = 0
        elif failed_count >= threshold * 3:
            print "Exit training, last entropy = %f, best entropy = %f" % (ent, best_entropy)
            break
    print "VALID ENT: %f, PPL: %f, BEST_ENT: %f, FAILED: %d" % (ent, 2 ** ent, best_entropy, failed_count)
    c += 1
    if c >= 20:
        break

timer.report()