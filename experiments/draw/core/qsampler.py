#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy import *


class Qsampler(NeuralLayer):
    def __init__(self, size, init=None):
        super(Qsampler, self).__init__("qsampler")

        self.prior_mean = 0.
        self.prior_log_sigma = 0.
        self.size = size
        self.init = init

    def setup(self):
        self.mean_transform = Dense(self.size, init=self.init).connect(self.input_dim)

        self.log_sigma_transform = Dense(self.size, init=self.init).connect(self.input_dim)

        self.register_inner_layers(self.mean_transform, self.log_sigma_transform)

    def sample(self, x, random_source):
        """
        Return a samples and the corresponding KL term

        Parameters:
            x

        Returns:
            z - tensor.matrix
                Samples drawn from Q(z|x)
            kl - tensor.vector
                KL(Q(z|x) || P_z)

        """
        mean = self.mean_transform.output(x)
        log_sigma = self.log_sigma_transform.output(x)

        # Sample from mean-zeros std.-one Gaussian
        z = mean + T.exp(log_sigma) * random_source

        # Calculate KL
        kl = (
            self.prior_log_sigma - log_sigma
            + 0.5 * (
                T.exp(2 * log_sigma) + (mean - self.prior_mean) ** 2
                ) / T.exp(2 * self.prior_log_sigma)
            - 0.5
        ).sum(axis=-1)

        return z, kl

    def sample_from_prior(self, random_source):
        """Sample z from the prior distribution P_z.
        Parameters
        ----------
        u : tensor.matrix
            gaussian random source
        Returns
        -------
        z : tensor.matrix
            samples
        """

        # Sample from mean-zeros std.-one Gaussian
        z = self.prior_mean + T.exp(self.prior_log_sigma) * random_source

        return z