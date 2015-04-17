import cPickle as pickle
import logging

from deepy import NetworkConfig, NeuralClassifier
import numpy as np

logging.basicConfig(level=logging.INFO)

data = pickle.load(open("/home/hadoop/data/eikaiwa/preprocessed/topic_distr_list.pkl"))
train_size = int(len(data)*0.8)
valid_size = len(data) - train_size
print "train valid sizes", train_size, valid_size
train_set, valid_set = [], []
train_source = data[:train_size]
valid_source = data[train_size: train_size + valid_size]

depth = 2

for source, target_set in [(train_source, train_set), (valid_source, valid_set)]:
    for i in xrange(len(source) - depth - 1):
        x = np.concatenate(source[i:i + depth])
        if len(x) != 500:
            continue
        y = source[i + depth]
        x -= np.mean(x)
        y -= np.mean(y)
        y = np.argmax(y)
        target_set.append(([x], [y]))


net_conf = NetworkConfig(input_size=500)
net_conf.layers = [NeuralLayer(size=50, activation='relu'), NeuralLayer(size=250, activation='softmax')]

network = NeuralClassifier(net_conf)
network.load_params("/home/hadoop/data/eikaiwa/topictracking_model.gz")


for d in valid_set:
    print d[1], network.classify(d[0])