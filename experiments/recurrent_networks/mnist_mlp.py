import time
import os
import logging

from deepy import NetworkConfig, TrainerConfig, NeuralClassifier, AdaGradTrainer
from deepy.networks import NeuralLayer
from deepy.dataset import MnistDataset

logging.basicConfig(level=logging.INFO)

mnist = (MnistDataset())

model_path = "/tmp/mnist_mlp_params.gz"

net_conf = NetworkConfig(input_size=28*28)
net_conf.layers = [NeuralLayer(size=300, activation="sigmoid"), NeuralLayer(size=10, activation='softmax')]

trainer_conf = TrainerConfig()
trainer_conf.learning_rate = 0.01
trainer_conf.weight_l2 = 0.0001
trainer_conf.hidden_l2 = 0.0001
trainer_conf.monitor_frequency = trainer_conf.validation_frequency = trainer_conf.test_frequency = 1
trainer_conf.test_frequency = 10

network = NeuralClassifier(net_conf)
trainer = AdaGradTrainer(network, config=trainer_conf)

if os.path.exists(model_path) and False:
    network.load_params(model_path)

start_time = time.time()
for k in list(trainer.train(mnist.train_set(), mnist.valid_set(), mnist.test_set())):
    pass
print k
end_time = time.time()

network.save_params(model_path)

print "time:", ((end_time - start_time )/ 60), "mins"