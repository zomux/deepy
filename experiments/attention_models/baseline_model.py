#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import numpy as np
from numpy import linalg as LA
from theano import tensor as T
import theano
from theano.tensor.shared_randomstreams import RandomStreams

from deepy import NeuralClassifier, NetworkConfig
from deepy.utils import build_activation, disconnected_grad
from deepy.utils.functions import FLOATX
from deepy.networks import NeuralLayer
from experiments.attention_models.gaussian_sampler import SampleMultivariateGaussian


class AttentionLayer(NeuralLayer):

    def __init__(self, activation='relu', std=0.1, disable_reinforce=False, random_glimpse=False):
        self.disable_reinforce = disable_reinforce
        self.random_glimpse = random_glimpse
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

        if not self.disable_reinforce:
            sampled_l_t = self._sample_gaussian(l_t, self.cov)
            sampled_pdf = self._multi_gaussian_pdf(disconnected_grad(sampled_l_t), l_t)
            wl_grad = T.grad(T.log(sampled_pdf), self.W_l)
        else:
            sampled_l_t = l_t
            wl_grad = self.W_l

        if self.random_glimpse and self.disable_reinforce:
            sampled_l_t = self.srng.uniform((2,)) * 0.8

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
        self._relu = build_activation("tanh")
        self._tanh = build_activation("tanh")
        self._softmax = build_activation("softmax")
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
        self.parameters = [self.W_l]


def get_network(model=None, std=0.005, disable_reinforce=False, random_glimpse=False):
    """
    Get baseline model.
    Parameters:
        model - model path
    Returns:
        network
    """
    network = NeuralClassifier(input_dim=28*28)
    network.stack_layer(AttentionLayer(std=std, disable_reinforce=disable_reinforce, random_glimpse=random_glimpse))
    if model and os.path.exists(model):
        network.load_params(model)
    return network

