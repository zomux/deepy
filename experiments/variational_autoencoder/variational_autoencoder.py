#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano.tensor as T
from deepy import NeuralLayer, AutoEncoder, Dense
from deepy import GaussianInitializer, global_theano_rand


class ReparameterizationLayer(NeuralLayer):
    """
    Reparameterization layer in a Variational encoder.
    Only binary output cost function is supported now.
    The prior value is recorded after the computation graph created.
    """

    def __init__(self, size):
        super(ReparameterizationLayer, self).__init__("VariationalEncoder")
        self.size = size
        self.output_dim = size
        self._prior = None

    def setup(self):
        self._mu_encoder = Dense(self.size, 'linear', init=GaussianInitializer(), random_bias=True).connect(self.input_dim)
        self._log_sigma_encoder = Dense(self.size, 'linear', init=GaussianInitializer(), random_bias=True).connect(self.input_dim)
        self.register_inner_layers(self._mu_encoder, self._log_sigma_encoder)

    def output(self, x):
        # Compute p(z|x)
        mu = self._mu_encoder.output(x)
        log_sigma = 0.5 * self._log_sigma_encoder.output(x)
        self._prior = 0.5* T.sum(1 + 2*log_sigma - mu**2 - T.exp(2*log_sigma))
        # Reparameterization
        eps = global_theano_rand.normal((x.shape[0], self.size))
        z = mu + T.exp(log_sigma) * eps
        return z

    def prior(self):
        """
        Get the prior value.
        """
        return self._prior


class VariationalAutoEncoder(AutoEncoder):
    """
    Variational Auto Encoder.
    Only binary output cost function is supported now.
    """

    def stack_reparameterization_layer(self, layer_size):
        """
        Perform reparameterization trick for latent variables.
        """
        self.rep_layer = ReparameterizationLayer(layer_size)
        self.stack_encoders(self.rep_layer)
        self._setup_monitors = True

    def _cost_func(self, y):
        logpxz  = - T.nnet.binary_crossentropy(y, self.input_variables[0]).sum()
        logp = logpxz + self.rep_layer.prior()
        # the lower bound is the mean value of logp
        cost = - logp
        if self._setup_monitors:
            self._setup_monitors = False
            self.training_monitors.append(("lower_bound", logp / y.shape[0]))
            self.testing_monitors.append(("lower_bound", logp / y.shape[0]))
        return cost
