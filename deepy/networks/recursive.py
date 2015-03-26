#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging as loggers

import theano
import theano.tensor as T

from deepy.util.functions import FLOATX
from deepy.util import build_activation
from deepy.networks.layer import NeuralLayer
from basic_nn import NeuralNetwork


logging = loggers.getLogger(__name__)


class RAELayer(NeuralLayer):

    def __init__(self, size, target_size=-1, activation='tanh', noise=0., dropouts=0.,
                 beta=0.0000001, optimization="ADAGRAD", unfolding=True, additional_h=True):
        """
        Binarized Recursive Layer
        Input:
        A sequence of terminal nodes in vectore representations.
        Output:
        Total error (euclidean distance)
        """
        super(RAELayer, self).__init__(size, activation, noise, dropouts)
        self.learning_rate = 0.1
        self.disable_bias = True
        self.target_size = target_size
        self.optimization = optimization
        self.beta = beta
        self.unfolding = unfolding
        self.additional_h = additional_h

    def connect(self, config, vars, x, input_n, id="UNKNOWN"):
        """
        Connect to a network
        :type config: deepy.conf.NetworkConfig
        :type vars: deepy.functions.VarMap
        :return:
        """
        self._config = config
        self._vars = vars
        self.input_n = input_n
        self.id = id
        self.x = x
        self._setup_params()
        self._setup_functions()
        self.connected = True

    def _recursive_step(self, i, p, x):
        x_t = x[i]
        # Encoding
        rep = self._activation_func(T.dot(p, self.W_e1) + T.dot(x_t, self.W_e2) + self.B_e)
        if self.unfolding:
            x_decs = self._unfold(rep, i)
            distance = T.sum((x_decs - x[: i + 1]) ** 2)
        else:
            # Decoding
            p_dec, x_dec = self._decode_step(rep)
            # Euclidean distance
            distance = T.sum((p_dec - p)**2 + (x_dec - x_t)**2)
        return rep, distance

    def _unfold(self, p, n):
        if self.additional_h:
            n += 1
        [ps, xs], _ = theano.scan(self._decode_step, outputs_info=[p, None], n_steps=n)
        if self.additional_h:
            return xs[::-1]
        else:
            return T.concatenate([xs, [ps[-1]]])[::-1]

    def _recursive_func(self):
        # Return total error
        if self.additional_h:
            h0 = self.h0
            start_index = 0
        else:
            h0 = self.x[0]
            start_index = 1
        [reps, distances], _ = theano.scan(self._recursive_step, sequences=[T.arange(start_index, self.x.shape[0])],
                                           outputs_info=[h0, None], non_sequences=[self.x])
        return reps[-1], T.sum(distances)

    def encode_func(self):
        if self.additional_h:
            h0 = self.h0
            start_index = 0
        else:
            h0 = self.x[0]
            start_index = 1
        [reps, _], _ = theano.scan(self._recursive_step, sequences=[T.arange(start_index, self.x.shape[0])],
                                           outputs_info=[h0, None], non_sequences=[self.x])
        return reps[-1]

    def _decode_step(self, p):
        p_dec = self._activation_func(T.dot(p, self.W_d1) + self.B_d1)
        x_dec = self._activation_func(T.dot(p, self.W_d2) + self.B_d2)
        return  p_dec, x_dec


    def decode_func(self):
        return self._unfold(self._vars.p, self._vars.n)

    def _setup_functions(self):
        self._assistive_params = []
        self._activation_func = build_activation(self.activation)
        self._softmax_func = build_activation('softmax')
        top_rep, self.output_func = self._recursive_func()
        # self.predict_func, self.predict_updates = self._encode_func()
        self.monitors.append(("top_rep<0.1", 100 * (abs(top_rep) < 0.1).mean()))
        self.monitors.append(("top_rep<0.9", 100 * (abs(top_rep) < 0.9).mean()))
        self.monitors.append(("top_rep:mean", abs(top_rep).mean()))

    def _setup_params(self):
        if self.target_size < 0:
            self.target_size = self.input_n

        self.W_e1 = self.create_weight(self.output_n, self.output_n, "enc1")
        self.W_e2 = self.create_weight(self.input_n, self.output_n, "enc2")
        self.B_e = self.create_bias(self.output_n, "enc")

        self.W_d1 = self.create_weight(self.output_n, self.output_n, "dec1")
        self.W_d2 = self.create_weight(self.output_n, self.input_n, "dec2")
        self.B_d1 = self.create_bias(self.output_n, "dec1")
        self.B_d2 = self.create_bias(self.input_n, "dec2")

        self.h0 = None
        if self.additional_h:
            self.h0 = self.create_vector(self.output_n, "h0")

        self.W = [self.W_e1, self.W_e2, self.W_d1, self.W_d2]
        self.B = [self.B_e, self.B_d1, self.B_d2]
        self.params = []

        # Just for decoding
        self._vars.n = T.iscalar("n")
        self._vars.p = T.vector("p", dtype=FLOATX)


class GeneralAutoEncoder(NeuralNetwork):

    def __init__(self, config):
        super(GeneralAutoEncoder, self).__init__(config)
        self._predict_compiled = False

    def setup_vars(self):
        super(GeneralAutoEncoder, self).setup_vars()

        # for a classifier, k specifies the correct labels for a given input.
        # No additional parameters

    @property
    def cost(self):
        return self.vars.y

    @property
    def monitors(self):
        for name, exp in self.special_monitors:
            yield name, exp

    def _compile(self):
        if not self._predict_compiled:
            rec_layer = self.layers[0]
            if hasattr(rec_layer, 'encode_inputs'):
                encode_inputs = rec_layer.encode_inputs
            else:
                encode_inputs = [self.vars.x]
            if hasattr(rec_layer, 'decode_inputs'):
                decode_inputs = rec_layer.decode_inputs
            else:
                decode_inputs = [self.vars.p, self.vars.n]
            self._encode_func = theano.function(encode_inputs, rec_layer.encode_func(), on_unused_input='warn')
            self._decode_func = theano.function(decode_inputs, rec_layer.decode_func(), on_unused_input='warn')
            self._predict_compiled = True

    def encode(self, *x):
        self._compile()
        return self._encode_func(*x)

    def decode(self, *x):
        self._compile()
        return self._decode_func(*x)


if __name__ == '__main__':
    pass