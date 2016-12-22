#!/usr/bin/env python
# -*- coding: utf-8 -*-

from layer import NeuralLayer
import deepy.tensor as T

class Attention(NeuralLayer):


    def __init__(self, hidden_size, input_dim=None):
        super(Attention, self).__init__("attention")
        self.input_dim = input_dim if input_dim else hidden_size
        self.hidden_size = hidden_size
        self._precomputed_inputs = None
        self.init(input_dim)

    def prepare(self):
        self.Ua = self.create_weight(self.input_dim, self.hidden_size, "ua")
        self.Wa = self.create_weight(self.hidden_size, self.hidden_size, "wa")
        self.Va = self.create_weight(label="va", shape=(self.hidden_size,))
        self.register_parameters(self.Va, self.Wa, self.Ua)


    def precompute(self, inputs):
        self._precomputed_inputs = T.dot(inputs, self.Ua)

    def compute_alignments(self, prev_state, inputs=None, mask=None):
        """
        Compute the alignment weights based on the previous state.
        """
        if not inputs and not self._precomputed_inputs:
            raise Exception("Either inputs or precomputed inputs shall be provided.")
        elif not self._precomputed_inputs:
            self.precompute(inputs)

        WaSp = T.dot(prev_state, self.Wa)
        UaH = self._precomputed_inputs
        # For test time the UaH will be (time, output_dim)
        if UaH.ndim == 2:
            preact = WaSp[:, None, :] + UaH[None, :, :]
        else:
            preact = WaSp[:, None, :] + UaH
        act = T.activate(preact, 'tanh')
        align_scores = T.dot(act, self.Va)  # ~ (batch, time)
        if mask:
            mask = (1 - mask) * -99.00
            if align_scores.ndim == 3:
                align_scores += mask[None, :]
            else:
                align_scores += mask
        align_weights = T.nnet.softmax(align_scores)
        return align_weights

    def compute_context_vector(self, prev_state, inputs=None, mask=None):
        """
        Compute the context vector with soft attention.
        """
        align_weights = self.compute_alignments(prev_state, inputs, mask)
        context_vector = T.sum(align_weights[:, :, None] * inputs, axis=1)
        return context_vector