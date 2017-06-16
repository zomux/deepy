#!/usr/bin/env python
# -*- coding: utf-8 -*-

from layer import NeuralLayer
import deepy.tensor as T

class Attention(NeuralLayer):

    AVAIL_ATT_TYPES = {'general', 'dot'}

    def __init__(self, input_dim=None, hidden_size=None, type='general'):
        super(Attention, self).__init__("attention")
        assert type in self.AVAIL_ATT_TYPES
        self.input_dim = input_dim if input_dim else 0
        self.hidden_size = hidden_size if hidden_size else input_dim
        self.type = type
        self.init(input_dim)

    def get_type(self):
        return self.type

    def prepare(self):
        if self.type == 'general':
            self.Ua = self.create_weight(self.input_dim, self.hidden_size, "ua")
            self.Wa = self.create_weight(self.hidden_size, self.hidden_size, "wa")
            self.Va = self.create_weight(label="va", shape=(self.hidden_size,))
            self.register_parameters(self.Va, self.Wa, self.Ua)


    def precompute(self, inputs):
        """
        Precompute partial values in the score function.
        """
        assert self.type == 'general'
        return T.dot(inputs, self.Ua)
    
    def compute_alignments_3d(self, queries, keys, mask=None):
        """
        Compute the alignment weights when states are 3d tensors.
        :param queries: tensor ~ (B, T2, H)
        :param keys: tensor ~ (B, T1, H)
        :param mask: matrix ~ (B, T1)
        :return: tensor ~ (B, T2, T1)
        """
        if self.type == 'dot':
            align_scores = T.sum(queries[:, :, None, :] * keys[:, None, :, :], axis=-1)
        elif self.type == 'general':
            WaSp = T.dot(queries, self.Wa)[:, :, None, :]
            UaH = keys[:, None, :, :]
            preact = WaSp + UaH
            act = T.tanh(preact)
            align_scores = T.dot(act, self.Va)
        else:
            raise NotImplementedError
        if mask:
            mask = (1 - mask) * -99.00
            align_scores += mask[:, None, :]
        ashp = align_scores.shape
        align_weights = T.nnet.softmax(align_scores.reshape((ashp[0] * ashp[1], ashp[2]))).reshape(ashp)
        return align_weights

    def compute_alignments(self, queries, keys, mask=None):
        """
        Compute the alignment weights based on the previous state.
        """
        if self.type == 'dot':
            if keys.ndim == 2:
                align_scores = T.sum(queries[:, None, :] * keys[None, :, :], axis=-1)
            else:
                align_scores = T.sum(queries[:, None, :] * keys, axis=-1)
        elif self.type == 'general':
            WaSp = T.dot(queries, self.Wa)
            UaH = keys
            # For test time the UaH will be (time, output_dim)
            if UaH.ndim == 2:
                preact = WaSp[:, None, :] + UaH[None, :, :]
            else:
                preact = WaSp[:, None, :] + UaH
            act = T.activate(preact, 'tanh')
            align_scores = T.dot(act, self.Va)  # ~ (batch, time)
        else:
            raise NotImplementedError
        if mask:
            mask = (1 - mask) * -99.00
            if align_scores.ndim == 3:
                align_scores += mask[None, :]
            else:
                align_scores += mask
        align_weights = T.nnet.softmax(align_scores)
        return align_weights

    def compute_context_vector(self, prev_state, input_values, precomputed_values=None, mask=None):
        """
        Compute the context vector with soft attention.
        """
        precomputed_values = precomputed_values if precomputed_values else self.precompute(input_values)
        align_weights = self.compute_alignments(prev_state, precomputed_values, mask)
        context_vector = T.sum(align_weights[:, :, None] * input_values, axis=1)
        return context_vector
