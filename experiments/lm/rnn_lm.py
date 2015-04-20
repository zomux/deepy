import time
import os
import sys
import logging

import numpy as np

from vocab import Vocab
from data_generator import LMDataGenerator
from deepy.conf import TrainerConfig
from deepy.trainers import MomentumTrainer
from deepy.layers import RNN


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

model_path = "/tmp/rnn_lm_params12.gz"
resource_dir = os.path.abspath(os.path.dirname(__file__)) + os.sep + "resources"

train_path = os.path.join(resource_dir, "ptb.train.txt")
train_small_path = os.path.join(resource_dir, "ptb.train.10k.txt")
valid_path = os.path.join(resource_dir, "ptb.valid.txt")
vocab = Vocab(char_based=True)
vocab.load(train_path, fixed_size=1000)



#
train_data = LMDataGenerator(vocab, train_small_path, target_vector=False, overlap=False,
                              history_len=9, _just_test=False, fixed_length=False, progress=True)
# valid_data = LMDataGenerator(vocab, question_valid, target_vector=False, overlap=False,
#                               history_len=9, _just_test=False, fixed_length=False, progress=False)
#
# trainer_conf = TrainerConfig()
# trainer_conf.learning_rate = 0.3
# trainer_conf.weight_l2 = 0.0001
# trainer_conf.hidden_l2 = 0.0001
# trainer_conf.monitor_frequency = trainer_conf.validation_frequency = trainer_conf.test_frequency = 1

# network = RecurrentNetwork()

# perplexity(network, valid_data)
# raise SystemExit
#
# trainer = MomentumTrainer(network, config=trainer_conf)
#
# start_time = time.time()
#
# best_entropy = 9999999.9
# if os.path.exists(model_path):
#     network.load_params(model_path)
#     # best_entropy = entropy(network, valid_data)
#
# def drytest(sent):
#     words = sent.split()
#     x = map(vocab.binvector, words)
#     y = network.classify(x)
#     network.clear_hidden()
#     return map(vocab.word, y)
#
# def expand(sent):
#     words = sent.split()
#     while words[-1] != "</s>":
#         x = map(vocab.binvector, words)
#         y = network.classify(x)
#         yw = vocab.word(y[-1])
#         words.append(yw)
#         network.clear_hidden()
#     return words
#
#
# # print expand("what do")
# # import pdb; pdb.set_trace()
#
# """
# Strategy:
# RESET -> 3 failures -> NON RESET
# 3 failures -> RELOAD
# 9 failures -> END
# """
# network.do_reset_grads = True
# threshold = 3
# failed_count = 0
# reload_count = 0
#
#
# for _ in trainer.train(train_data):
#     print ""
#     ent = entropy(network, valid_data)
#     if ent < best_entropy:
#         best_entropy = ent
#         failed_count = 0
#         reload_count = 0
#         network.save_params(model_path)
#     else:
#         failed_count += 1
#         reload_count += 1
#         if network.do_reset_grads and failed_count >= threshold:
#             print "Turn off resetting gradient velocities"
#             network.load_params(model_path)
#             network.do_reset_grads = False
#             failed_count = 0
#             reload_count = 0
#         elif reload_count >= threshold:
#             network.load_params(model_path)
#             reload_count = 0
#         elif failed_count >= threshold * 3:
#             print "Exit training, last entropy = %f, best entropy = %f" % (ent, best_entropy)
#             break
#     print "VALID ENT: %f, PPL: %f, BEST_ENT: %f, FAILED: %d" % (ent, 2 ** ent, best_entropy, failed_count)
#
# end_time = time.time()
#
#
# print "elapsed time:", (end_time - start_time) / 60, "mins"
