import math
import time
import os
import logging

from deepy.dataset import AbstractDataset
from nlpy.util import FeatureContainer
import theano.tensor as T
from deepy import NetworkConfig, TrainerConfig, SGDTrainer
from deepy.networks import NeuralLayer
from deepy.networks.basic_nn import NeuralNetwork
from deepy.functions import monitor_var
from deepy import nnprocessors


logging.basicConfig(level=logging.INFO)
import random

class CompDataset(AbstractDataset):

    def __init__(self, target_format=None):
        super(CompDataset, self).__init__(target_format)
        self._train_set = lambda: FeatureContainer(path="/home/hadoop/personal/comp1/data/subtrain.svmdata.scaled", feature_n=930)
        self._valid_set = lambda: FeatureContainer(path="/home/hadoop/personal/comp1/data/subvalid.svmdata.scaled", feature_n=930)
        self._test_set = lambda: FeatureContainer(path="/home/hadoop/personal/comp1/data/subtest.svmdata.scaled", feature_n=930)
        self._target_size = 2
        self.ts = None

    def _target_map(self, g, is_test=True):
        penalty_bound = 5.0
        xs = []
        ys = []
        for item in g:
            if is_test:
                penalty = 0
            else:
                penalty = penalty_bound * (1.0 - (float(item[2]) / 1800000))
            xs.append(item[0])
            if item[1] == -1:
                 ys.append([0, penalty])
            else:
                ys.append([1, penalty])
            if len(xs) == 1000:
                yield xs, ys
                xs = []
                ys = []
        if xs:
            yield xs, ys

    def train_set(self):
        if not self.ts:
            self.ts = []
            c = 1
            for item in self._train_set().read():
                self.ts.append((item[0], item[1], c))
                c += 1
        random.shuffle(self.ts)
        return self._target_map(self.ts, is_test=False)

    def valid_set(self):
        return self._target_map(self._valid_set().read())

    def test_set(self):
        return self._target_map(self._test_set().read())

class FakeGenerator(object):

  def __init__(self, dataset, method_name):
    self.dataset = dataset
    self.method_name = method_name

  def __iter__(self):
    return getattr(self.dataset, self.method_name)()


data = CompDataset()
DISABLE_MONITORING = True

# for d, y in (data.test_set()):
#     print np.min(d), np.min(y)
# raise SystemExit

def log_loss_func(self):
    epsilon = 1e-15
    inverse_epsilon = (1. - 1e-6)
    y = monitor_var(self.vars.y[:, 0], "vars.y", disabled=DISABLE_MONITORING)
    y =  y * (y >= epsilon) + (y < epsilon) * epsilon
    y =  y * (y <= inverse_epsilon) + (y > inverse_epsilon) * inverse_epsilon
    y = monitor_var(y, "y", disabled=DISABLE_MONITORING)
    k = monitor_var(self.vars.k, "k", disabled=DISABLE_MONITORING)
    loss = monitor_var(k * T.log(y) + (1 - k) * T.log(1 - y), "loss", disabled=DISABLE_MONITORING)
    mean = monitor_var(-T.mean(loss), "mean", disabled=DISABLE_MONITORING)
    return mean

def log_loss_func_softmax(self):
    y1 = self.vars.y[:, 0]
    y2 = 1 - y1
    loss = self.vars.k[:, 0] * T.log(y1) + (1 - self.vars.k[:, 0]) * T.log(y2)
    return -T.mean(loss)

class SimpleRegressor(NeuralNetwork):
    '''A regressor attempts to produce a target output.'''

    def setup_vars(self):
        super(SimpleRegressor, self).setup_vars()

        # the k variable holds the target output for input x.
        self.vars.k = T.matrix('k')
        self.inputs.append(self.vars.k)

    @property
    def cost(self):
        # err = self.vars.y[:, 0] - self.vars.k
        # return T.mean(err * err)
        y1 = self.vars.y[:, 0]
        y2 = 1 - y1
        penalty_vector = 0.9 ** self.vars.k[:, 1]
        loss = (self.vars.k[:, 0] * T.log(y1) + (1 - self.vars.k[:, 0]) * T.log(y2)) * penalty_vector
        return -T.mean(loss)

        # return T.mean(self.vars.k * T.log(self.vars.y[:, 0]) + (1 - self.vars.k) * T.log((1-self.vars.y[:, 0])))

        # epsilon = 1e-15
        # inverse_epsilon = (1. - 1e-6)
        # y = theano_monitor_var(self.vars.y[:, 0], "vars.y", disabled=DISABLE_MONITORING)
        # y *= self.vars.k
        # y =  y * (y >= epsilon) + (y < epsilon) * epsilon
        # y =  y * (y <= inverse_epsilon) + (y > inverse_epsilon) * inverse_epsilon
        # return -( T.sum(T.log(y)) / self.vars.y.shape[0] )

    @property
    def loss(self):
        return log_loss_func_softmax(self)

    @property
    def monitors(self):
        yield 'loss', self.loss
        for i, h in enumerate(self.hiddens):
            yield 'h{}<0.1'.format(i+1), 100 * (abs(h) < 0.1).mean()
            yield 'h{}<0.9'.format(i+1), 100 * (abs(h) < 0.9).mean()
        for name, exp in self.special_monitors:
            yield name, exp


def log_loss(preds, ys):
    epsilon = 1e-15
    loss = []
    for pred, y in zip(preds, ys):
        pred = max(epsilon, pred)
        pred = min(1-epsilon, pred)
        l = y * math.log(pred) + (1 - y) * math.log(1 - pred)
        loss.append(-l)
    return loss

class CustomizedInputLayer(NeuralLayer):

    def _setup_functions(self):
        if self.shared_bias:
            self._vars.update_if_not_existing(self.shared_bias, self.B)
        bias = self.B1 if not self.shared_bias else self._vars.get(self.shared_bias)
        if self.disable_bias:
            bias = 0

        self._activation_func = nnprocessors.build_activation(self.activation)
        self.preact_func = T.dot(self.L1.output_func, self.W1) + T.dot(self.L2.output_func, self.W2) \
                           + T.dot(self.L3.output_func, self.W3) + bias
        self.output_func = nnprocessors.add_noise(
                self._activation_func(self.preact_func),
                self.noise,
                self.dropouts)

    def _setup_params(self):
        self.L1 = NeuralLayer(size=300, activation="relu", disable_bias=False, dropouts=0.3)
        self.L2 = NeuralLayer(size=300, activation="relu", disable_bias=False, dropouts=0.3)
        self.L3 = NeuralLayer(size=500, activation="relu", disable_bias=False, dropouts=0.3)
        self.L1.connect(self._config, self._vars, self.x[:, 0:13], 13, "SCAT_LA_1")
        self.L2.connect(self._config, self._vars, self.x[:, 13: 13 + 205], 205, "SCAT_LA_2")
        self.L3.connect(self._config, self._vars, self.x[:, 13 + 205:13 + 205 + 712], 712, "SCAT_LA_3")
        self.W1, self.B1, p1 = self.create_params(300, self.output_n, "SCAT_W_1")
        self.W2, _, p2 = self.create_params(300, self.output_n, "SCAT_W_2")
        self.W3, _, p3 = self.create_params(500, self.output_n, "SCAT_W_3")
        self.param_count = p1 + p2 + p3 + self.L1.param_count + self.L2.param_count + self.L3.param_count
        self.W = [self.W1, self.W2, self.W3] + [self.L1.W , self.L2.W , self.L3.W]
        self.B = [self.B1, self.L1.B, self.L2.B, self.L3.B]


# -----------

model_path = "/tmp/click_mlp_params_6_scat_sigtanh_h.gz"

net_conf = NetworkConfig(input_size=930)
net_conf.layers = [#CustomizedInputLayer(size=200, activation="relu", disable_bias=False),
                   #NeuralLayer(size=100, activation="relu", disable_bias=False),
                   CustomizedInputLayer(size=1, activation='sigmoid', disable_bias=False)]
net_conf.input_dropouts = 0.1

trainer_conf = TrainerConfig()
trainer_conf.learning_rate = 0.01
trainer_conf.weight_l2 = 0.0001
trainer_conf.hidden_l2 = 0.0001
trainer_conf.monitor_frequency = trainer_conf.validation_frequency = trainer_conf.test_frequency = 1

network = SimpleRegressor(net_conf)

trainer = SGDTrainer(network, config=trainer_conf)

if os.path.exists(model_path):
    network.load_params(model_path)

# "predict"
# losses = []
# for xs, ys in data.test_set():
#     loss = log_loss(network.predict(xs)[:, 0], ys)
#     losses.extend(loss)
# print np.mean(np.array(losses))
# raise SystemExit

start_time = time.time()
for k in list(trainer.train(FakeGenerator(data, "train_set"), FakeGenerator(data, "valid_set"), FakeGenerator(data, "test_set"))):
    pass
print k
end_time = time.time()

network.save_params(model_path)

print "time:", ((end_time - start_time )/ 60), "mins"
