#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

import numpy as np
from numpy import linalg as LA
from theano import tensor as T
import theano

from deepy.utils.functions import FLOATX
from deepy.trainers import CustomizeTrainer
from deepy.trainers.optimize import optimize_function


class AttentionTrainer(CustomizeTrainer):

    def __init__(self, network, attention_layer, config):
        """
        Parameters:
            network - AttentionNetwork
            config - training config
        :type network: NeuralClassifier
        :type attention_layer: experiments.attention_models.baseline_model.AttentionLayer
        :type config: TrainerConfig
        """
        super(AttentionTrainer, self).__init__(network, config)
        self.large_cov_mode = False
        self.batch_size = config.get("batch_size", 20)
        self.disable_backprop = config.get("disable_backprop", False)
        self.disable_reinforce = config.get("disable_reinforce", False)
        self.last_average_reward = 999
        self.turn = 1
        self.layer = attention_layer
        if self.disable_backprop:
            grads = []
        else:
            grads = [T.grad(self.cost, p) for p in network.weights + network.biases]
        if self.disable_reinforce:
            grad_l = self.layer.W_l
        else:
            grad_l = self.layer.wl_grad
        self.batch_wl_grad = np.zeros(attention_layer.W_l.get_value().shape, dtype=FLOATX)
        self.batch_grad = [np.zeros(p.get_value().shape, dtype=FLOATX) for p in network.weights + network.biases]
        self.grad_func = theano.function(network.inputs, [self.cost, grad_l, attention_layer.positions, attention_layer.last_decision] + grads, allow_input_downcast=True)
        self.opt_interface = optimize_function(self.network.weights + self.network.biases, self.config)
        self.l_opt_interface = optimize_function([self.layer.W_l], self.config)
        # self.opt_interface = gradient_interface_future(self.network.weights + self.network.biases, config=self.config)
        # self.l_opt_interface = gradient_interface_future([self.layer.W_l], config=self.config)

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

