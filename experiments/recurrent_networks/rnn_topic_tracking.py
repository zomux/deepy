import time
import cPickle as pickle
import logging

from deepy import NetworkConfig, TrainerConfig, NeuralClassifier, SGDTrainer
from deepy.networks import NeuralLayer
import numpy as np
logging.basicConfig(level=logging.INFO)

data = pickle.load(open("/home/hadoop/data/eikaiwa/preprocessed/topic_distr_list.pkl"))
train_size = int(len(data)*0.8)
valid_size = len(data) - train_size
print "train valid sizes", train_size, valid_size
train_set, valid_set = [], []
train_source = data[:train_size]
valid_source = data[train_size: train_size + valid_size]

depth = 1

for source, target_set in [(train_source, train_set), (valid_source, valid_set)]:
    batch_x, batch_y = [], []
    for i in xrange(len(source) - depth - 1):
        x = np.concatenate(source[i:i + depth])
        if len(x) != 250:
            continue
        y = source[i + depth]
        x -= np.mean(x)
        y -= np.mean(y)
        y = np.argmax(x)
        batch_x.append(x)
        batch_y.append(y)
        if len(batch_x) >= 1:
            target_set.append((batch_x, batch_y))
            batch_x, batch_y = [], []

net_conf = NetworkConfig(input_size=250)
net_conf.layers = [NeuralLayer(size=250, activation='softmax')]

trainer_conf = TrainerConfig()
trainer_conf.learning_rate = 0.01
trainer_conf.weight_l2 = 0.0001
trainer_conf.hidden_l2 = 0.0001
trainer_conf.monitor_frequency = trainer_conf.validation_frequency = trainer_conf.test_frequency = 1

network = NeuralClassifier(net_conf)
trainer = SGDTrainer(network, config=trainer_conf)

start_time = time.time()
for k in list(trainer.train(train_set, valid_set)):
    pass
print k
end_time = time.time()
network.save_params("/tmp/topictracking.gz")