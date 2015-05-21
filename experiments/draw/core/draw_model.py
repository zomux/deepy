#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy import *
from draw_layer import DrawLayer

class DrawModel(NeuralNetwork):

    def __init__(self, image_width, image_height, attention_times, config=None):
        super(DrawModel, self).__init__(image_width * image_height, config=config)
        self.stack(DrawLayer(image_width, image_height, attention_times))

    @property
    def cost(self):
        """
        The output of DrawLayer is cost.
        """
        return self.output

    @property
    def test_cost(self):
        """
        The output of DrawLayer is cost.
        """
        return self.test_output