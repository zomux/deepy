#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import RNN
from deepy.utils import GaussianInitializer, IdentityInitializer, FLOATX

MAX_IDENTITY_VALUE = 0.99
MIN_IDENTITY_VALUE = 0.0

class IRNN(RNN):
    """
    The implementation of http://arxiv.org/abs/1504.00941 .
    RNN with weight initialization using identity matrix.
    """

    def __init__(self, hidden_size, input_type="sequence", output_type="one", steps=None, go_backwards=False,
                 weight_scale=0.9, bound_recurrent_weight=True, mask=None,
                 persistent_state=False, reset_state_for_input=None, batch_size=None,
                 second_input=None, second_input_size=None):
        super(IRNN, self).__init__(hidden_size,
                                   input_type=input_type, output_type=output_type,
                                   hidden_activation="relu", steps=steps,
                                   hidden_init=IdentityInitializer(scale=weight_scale),
                                   input_init=GaussianInitializer(deviation=0.001),
                                   persistent_state=persistent_state, reset_state_for_input=reset_state_for_input,
                                   batch_size=batch_size, go_backwards=go_backwards,
                                   mask=mask, second_input=second_input, second_input_size=second_input_size)
        self.name = "irnn"
        if bound_recurrent_weight:
            self.register_training_callbacks(self.training_callback)

    def training_callback(self):
        w_value = self.W_h.get_value(borrow=True)
        changed = False
        if w_value.max() > MAX_IDENTITY_VALUE:
            w_value = w_value * (w_value <= MAX_IDENTITY_VALUE) + MAX_IDENTITY_VALUE * (w_value > MAX_IDENTITY_VALUE)
            changed = True
        if w_value.min() < MIN_IDENTITY_VALUE:
            w_value = w_value * (w_value >= MIN_IDENTITY_VALUE) + MIN_IDENTITY_VALUE * (w_value < MIN_IDENTITY_VALUE)
            changed = True
        if changed:
            self.W_h.set_value(w_value.astype(FLOATX))
