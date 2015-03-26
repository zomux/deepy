import time
import os

import theano.tensor as T

from deepy import NetworkConfig, TrainerConfig, NeuralRegressor
from deepy.trainers.trainer import AdamTrainer
from deepy.networks import NeuralLayer
from deepy.dataset import MnistDataset, MiniBatches
from deepy.util import build_activation



########

class CropLayer(NeuralLayer):

    def __init__(self, activation='relu'):
        super(CropLayer, self).__init__(128, activation)

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
        x = self.x[:, 2:]
        l = self.x[:, :2]
        layer1_x = self._activation_func(T.dot(x, self.W_x))
        layer1_l = self._activation_func(T.dot(l, self.W_l))
        layer2 = self._activation_func(T.dot(layer1_x, self.W_hx) * T.dot(layer1_l, self.W_hl) )
        return layer2

    def _setup_functions(self):
        self._assistive_params = []
        self._activation_func = build_activation(self.activation)
        self.output_func = self._output_func()

    def _setup_params(self):
        self.W_x = self.create_weight(28*28, 128, suffix="x")
        self.W_l = self.create_weight(2, 128, suffix="l")
        self.W_hx = self.create_weight(128, 128, suffix="hx")
        self.W_hl = self.create_weight(128, 128, suffix="hl")


        self.W = [self.W_x, self.W_l]
        self.B = []
        self.params = []
########

import logging
logging.basicConfig(level=logging.INFO)

mnist = MiniBatches(MnistDataset())

model_path = "/tmp/crop1_model.gz"

net_conf = NetworkConfig(input_size=28*28+2)
net_conf.layers = [CropLayer(), NeuralLayer(size=8*8, activation='relu')]

trainer_conf = TrainerConfig()
trainer_conf.learning_rate = 0.01
trainer_conf.weight_l2 = 0.0001
trainer_conf.hidden_l2 = 0.0001
trainer_conf.monitor_frequency = trainer_conf.validation_frequency = trainer_conf.test_frequency = 1
trainer_conf.test_frequency = 10

network = NeuralRegressor(net_conf)
trainer = AdamTrainer(network, config=trainer_conf)

if os.path.exists(model_path) and False:
    network.load_params(model_path)

start_time = time.time()
for k in list(trainer.train(mnist.train_set(), mnist.valid_set(), mnist.test_set())):
    pass
print k
end_time = time.time()

network.save_params(model_path)

print "time:", ((end_time - start_time )/ 60), "mins"