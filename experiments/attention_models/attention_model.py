import time
import os
import sys
import logging

from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import numpy.linalg as LA

from deepy import NetworkConfig, TrainerConfig, NeuralClassifier
from deepy.networks import NeuralLayer
from deepy.dataset import MnistDataset, MiniBatches
from deepy import nnprocessors
from deepy.functions import FLOATX, disconnected_grad
from deepy.trainers.optimize import gradient_interface
from experiments.attention_models.gaussian_sampler import SampleMultivariateGaussian


logging.basicConfig(level=logging.INFO)
import theano
import theano.tensor as T
from deepy.trainers import CustomizeTrainer

class AttentionLayer(NeuralLayer):

    def __init__(self, activation='relu', std=0.1):
        self.gaussian_std = std
        super(AttentionLayer, self).__init__(10, activation)

    def connect(self, config, vars, x, input_n, id="UNKNOWN"):
        self._config = config
        self._vars = vars
        self.input_n = input_n
        self.id = id
        self.x = x
        self._setup_params()
        self._setup_functions()
        self.connected = True

    def _glimpse_sensor(self, x_t, l_p):
        """
        Parameters:
            x_t - 28x28 image
            l_p - 2x1 focus vector
        Returns:
            4x12 matrix
        """
        # Turn l_p to the left-top point of rectangle
        l_p = l_p * 14 + 14 - 2
        l_p = T.cast(T.round(l_p), "int32")

        l_p = l_p * (l_p >= 0)
        l_p = l_p * (l_p < 24) + (l_p >= 24) * 23
        l_p2 = l_p - 2
        l_p2 = l_p2 * (l_p2 >= 0)
        l_p2 = l_p2 * (l_p2 < 20) + (l_p2 >= 20) * 19
        l_p3 = l_p - 6
        l_p3 = l_p3 * (l_p3 >= 0)
        l_p3 = l_p3 * (l_p3 < 16) + (l_p3 >= 16) * 15
        glimpse_1 = x_t[l_p[0]: l_p[0] + 4][:, l_p[1]: l_p[1] + 4]
        glimpse_2 = x_t[l_p2[0]: l_p2[0] + 8][:, l_p2[1]: l_p2[1] + 8]
        glimpse_2 = theano.tensor.signal.downsample.max_pool_2d(glimpse_2, (2,2))
        glimpse_3 = x_t[l_p3[0]: l_p3[0] + 16][:, l_p3[1]: l_p3[1] + 16]
        glimpse_3 = theano.tensor.signal.downsample.max_pool_2d(glimpse_3, (4,4))
        return T.concatenate([glimpse_1, glimpse_2, glimpse_3])

    def _refined_glimpse_sensor(self, x_t, l_p):
        """
        Parameters:
            x_t - 28x28 image
            l_p - 2x1 focus vector
        Returns:
            7*14 matrix
        """
        # Turn l_p to the left-top point of rectangle
        l_p = l_p * 14 + 14 - 4
        l_p = T.cast(T.round(l_p), "int32")

        l_p = l_p * (l_p >= 0)
        l_p = l_p * (l_p < 21) + (l_p >= 21) * 20
        glimpse_1 = x_t[l_p[0]: l_p[0] + 7][:, l_p[1]: l_p[1] + 7]
        # glimpse_2 = theano.tensor.signal.downsample.max_pool_2d(x_t, (4,4))
        # return T.concatenate([glimpse_1, glimpse_2])
        return glimpse_1

    def _multi_gaussian_pdf(self, vec, mean):
        norm2d_var = ((1.0 / T.sqrt((2*np.pi)**2 * self.cov_det_var)) *
                      T.exp(-0.5 * ((vec-mean).T.dot(self.cov_inv_var).dot(vec-mean))))
        return norm2d_var

    def _glimpse_network(self, x_t, l_p):
        """
        Parameters:
            x_t - 28x28 image
            l_p - 2x1 focus vector
        Returns:
            4x12 matrix
        """
        sensor_output = self._refined_glimpse_sensor(x_t, l_p)
        sensor_output = T.flatten(sensor_output)
        h_g = self._relu(T.dot(sensor_output, self.W_g0))
        h_l = self._relu(T.dot(l_p, self.W_g1))
        g = self._relu(T.dot(h_g, self.W_g2_hg) + T.dot(h_l, self.W_g2_hl))
        return g

    def _location_network(self, h_t):
        """
        Parameters:
            h_t - 256x1 vector
        Returns:
            2x1 focus vector
        """
        return T.dot(h_t, self.W_l)

    def _action_network(self, h_t):
        """
        Parameters:
            h_t - 256x1 vector
        Returns:
            10x1 vector
        """
        z = self._relu(T.dot(h_t, self.W_a) + self.B_a)
        return self._softmax(z)

    def _core_network(self, l_p, h_p, x_t):
        """
        Parameters:
            x_t - 28x28 image
            l_p - 2x1 focus vector
            h_p - 256x1 vector
        Returns:
            h_t, 256x1 vector
        """
        g_t = self._glimpse_network(x_t, l_p)
        h_t = self._tanh(T.dot(g_t, self.W_h_g) + T.dot(h_p, self.W_h) + self.B_h)
        l_t = self._location_network(h_t)
        sampled_l_t = self._sample_gaussian(l_t, self.cov)
        sampled_pdf = self._multi_gaussian_pdf(disconnected_grad(sampled_l_t), l_t)
        # sampled_l_t = self.srng.uniform((2,)) * 0.8

        wl_grad = T.grad(T.log(sampled_pdf), self.W_l)
        # wl_grad = self.W_l

        a_t = self._action_network(h_t)

        return sampled_l_t, h_t, a_t, wl_grad


    def _output_func(self):
        self.x = self.x.reshape((28, 28))
        [l_ts, h_ts, a_ts, wl_grads], _ = theano.scan(fn=self._core_network,
                         outputs_info=[self.l0, self.h0, None, None],
                         non_sequences=[self.x],
                         n_steps=5)

        self.positions = l_ts
        self.last_decision = T.argmax(a_ts[-1])
        wl_grad = T.sum(wl_grads, axis=0) / wl_grads.shape[0]
        self.wl_grad = wl_grad
        return a_ts[-1].reshape((1,10))

    def _setup_functions(self):
        self._assistive_params = []
        self._relu = nnprocessors.build_activation("tanh")
        self._tanh = nnprocessors.build_activation("tanh")
        self._softmax = nnprocessors.build_activation("softmax")
        self.output_func = self._output_func()

    def _setup_params(self):
        self.srng = RandomStreams(seed=234)
        self.large_cov = np.array([[0.06,0],[0,0.06]], dtype=FLOATX)
        self.small_cov = np.array([[self.gaussian_std,0],[0,self.gaussian_std]], dtype=FLOATX)
        self.cov = theano.shared(np.array(self.small_cov, dtype=FLOATX))
        self.cov_inv_var = theano.shared(np.array(LA.inv(self.small_cov), dtype=FLOATX))
        self.cov_det_var = theano.shared(np.array(LA.det(self.small_cov), dtype=FLOATX))
        self._sample_gaussian = SampleMultivariateGaussian()

        self.W_g0 = self.create_weight(7*7, 128, suffix="g0")
        self.W_g1 = self.create_weight(2, 128, suffix="g1")
        self.W_g2_hg = self.create_weight(128, 256, suffix="g2_hg")
        self.W_g2_hl = self.create_weight(128, 256, suffix="g2_hl")

        self.W_h_g = self.create_weight(256, 256, suffix="h_g")
        self.W_h = self.create_weight(256, 256, suffix="h")
        self.B_h = self.create_bias(256, suffix="h")
        self.h0 = self.create_vector(256, "h0")
        self.l0 = self.create_vector(2, "l0")
        self.l0.set_value(np.array([-1, -1], dtype=FLOATX))

        self.W_l = self.create_weight(256, 2, suffix="l")
        self.W_l.set_value(self.W_l.get_value() / 10)
        self.B_l = self.create_bias(2, suffix="l")
        self.W_a = self.create_weight(256, 10, suffix="a")
        self.B_a = self.create_bias(10, suffix="a")


        self.W = [self.W_g0, self.W_g1, self.W_g2_hg, self.W_g2_hl, self.W_h_g, self.W_h, self.W_a]
        self.B = [self.B_h, self.B_a]
        self.params = [self.W_l]

############

class AttentionTrainer(CustomizeTrainer):

    def __init__(self, network, attention_layer, config, batch_size=20, disable_backprop=False, disable_rienforce=False):
        """
        Parameters:
            network - AttentionNetwork
            config - training config
        :type network: NeuralClassifier
        :type attention_layer: AttentionLayer
        :type config: TrainerConfig
        """
        super(AttentionTrainer, self).__init__(network, config)
        self.disable_backprop = disable_backprop
        self.disable_reinforce = disable_rienforce
        self.large_cov_mode = False
        self.batch_size = 20
        self.last_average_reward = 999
        self.turn = 1
        self.layer = attention_layer
        if self.disable_backprop:
            grads = []
        else:
            grads = [T.grad(self.J, p) for p in network.weights + network.biases]
        if self.disable_reinforce:
            grad_l = self.layer.W_l
        else:
            grad_l = self.layer.wl_grad
        self.batch_wl_grad = np.zeros(attention_layer.W_l.get_value().shape, dtype=FLOATX)
        self.batch_grad = [np.zeros(p.get_value().shape, dtype=FLOATX) for p in network.weights + network.biases]
        self.grad_func = theano.function(network.inputs, [self.J, grad_l, attention_layer.positions, attention_layer.last_decision] + grads, allow_input_downcast=True)
        self.opt_interface = gradient_interface(self.network.weights + self.network.biases, lr=self.config.learning_rate, method="ADAGRAD", gsum_regularization=0.0001)
        self.l_opt_interface = gradient_interface([self.layer.W_l], lr=self.config.learning_rate, method="ADAGRAD", max_norm=0.8, gsum_regularization=0.0001)


    def update_parameters(self, update_wl):
        if not self.disable_backprop:
            grads = [self.batch_grad[i] / self.batch_size for i in range(len(self.network.weights + self.network.biases))]
            self.opt_interface(*grads)
        # REINFORCE update
        if update_wl and not self.disable_reinforce:
            if np.sum(self.batch_wl_grad) == 0:
                sys.stdout.write("[0 WLG] ")
                sys.stdout.flush()
            else:
                grad_wl = self.batch_wl_grad / self.batch_size
                self.l_opt_interface(grad_wl)

    def train_func(self, train_set):
        cost_sum = 0.0
        batch_cost = 0.0
        counter = 0
        total = 0
        total_reward = 0
        batch_reward = 0
        total_position_value = 0
        pena_count = 0
        for d in train_set:
            pairs = self.grad_func(*d)
            cost = pairs[0]
            if cost > 10 or np.isnan(cost):
                sys.stdout.write("X")
                sys.stdout.flush()
                continue
            batch_cost += cost

            wl_grad = pairs[1]
            max_position_value = np.max(np.absolute(pairs[2]))
            total_position_value += max_position_value
            last_decision = pairs[3]
            target_decision = d[1][0]
            reward = 0.005 if last_decision == target_decision else 0
            if max_position_value > 0.8:
                reward =  0
            total_reward += reward
            batch_reward += reward
            if self.last_average_reward == 999 and total > 2000:
                self.last_average_reward = total_reward / total
            if not self.disable_reinforce:
                self.batch_wl_grad += wl_grad *  - (reward - self.last_average_reward)
            if not self.disable_backprop:
                for grad_cache, grad in zip(self.batch_grad, pairs[4:]):
                    grad_cache += grad
            counter += 1
            total += 1
            if counter >= self.batch_size:
                if total == counter: counter -= 1
                self.update_parameters(self.last_average_reward < 999)

                # Clean batch gradients
                if not self.disable_reinforce:
                    self.batch_wl_grad *= 0
                if not self.disable_backprop:
                    for grad_cache in self.batch_grad:
                        grad_cache *= 0

                if total % 1000 == 0:
                    sys.stdout.write(".")
                    sys.stdout.flush()

                # Cov
                if not self.disable_reinforce:
                    cov_changed = False
                    if batch_reward / self.batch_size < 0.001:
                        if not self.large_cov_mode:
                            if pena_count > 20:
                                self.layer.cov.set_value(self.layer.large_cov)
                                print "[LCOV]",
                                cov_changed = True
                            else:
                                pena_count += 1
                        else:
                            pena_count = 0
                    else:
                        if self.large_cov_mode:
                            if pena_count > 20:
                                self.layer.cov.set_value(self.layer.small_cov)
                                print "[SCOV]",
                                cov_changed = True
                            else:
                                pena_count += 1
                        else:
                            pena_count = 0
                    if cov_changed:
                        self.large_cov_mode = not self.large_cov_mode
                        self.layer.cov_inv_var.set_value(np.array(LA.inv(self.layer.cov.get_value()), dtype=FLOATX))
                        self.layer.cov_det_var.set_value(LA.det(self.layer.cov.get_value()))

                # Clean batch cost
                counter = 0
                cost_sum += batch_cost
                batch_cost = 0.0
                batch_reward = 0
        if total == 0:
            return "COST OVERFLOW"

        sys.stdout.write("\n")
        self.last_average_reward = (total_reward / total)
        self.turn += 1
        return "J: %.2f, Avg R: %.4f, Avg P: %.2f" % ((cost_sum / total), self.last_average_reward, (total_position_value / total))

############

mnist = MiniBatches((MnistDataset()), batch_size=1)

model_path = "/tmp/mnist_att_params2.gz"

net_conf = NetworkConfig(input_size=28*28)
attention_layer = AttentionLayer(std=0.005)
net_conf.layers = [attention_layer]

trainer_conf = TrainerConfig()
trainer_conf.learning_rate = 0.005
trainer_conf.weight_l2 = 0.0001
trainer_conf.hidden_l2 = 0.0001
trainer_conf.monitor_frequency = trainer_conf.validation_frequency = trainer_conf.test_frequency = 1
trainer_conf.test_frequency = 10
trainer_conf.patience = 100

network = NeuralClassifier(net_conf)
network.inputs[0].tag.test_value = mnist.valid_set()[0][0]
network.inputs[1].tag.test_value = mnist.valid_set()[0][1]


if os.path.exists(model_path) and True:
    network.load_params(model_path)
    # import pdb;pdb.set_trace()
    # sys.exit()

trainer = AttentionTrainer(network, attention_layer, config=trainer_conf)
# trainer = AdaDeltaTrainer(network, config=trainer_conf)

start_time = time.time()
c = 1
for k in list(trainer.train(mnist.train_set(), mnist.valid_set(), mnist.test_set())):
    if c > 10:
        break
    c += 1
print k
end_time = time.time()

network.save_params(model_path)

print "time:", ((end_time - start_time )/ 60), "mins"
