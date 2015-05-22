#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy import *
from attention import *

class AttentionWriter(object):

    def __init__(self, input_dim, img_width, img_height, glimpse_size, init=None):
        """
        Parameters:
            input_dim - dim of h_dec
        :param input_dim:
        :param img_width:
        :param img_height:
        :param glimpse_size:
        :return:
        """

        self.img_width = img_width
        self.img_height = img_height
        self.glimpse_size = glimpse_size
        self.input_dim = input_dim

        self.zoomer = ZoomableAttentionWindow(self.img_height, self.img_width, self.glimpse_size)

        self.director_model = Chain(self.input_dim).stack(Dense(5, init=init))
        self.decoding_model = Chain(self.input_dim).stack(Dense(self.glimpse_size*self.glimpse_size, init=init))


    def write(self, h_dec):
        glimpse = self.decoding_model.output(h_dec)
        director_output = self.director_model.output(h_dec)

        center_y, center_x, delta, sigma, gamma = self.zoomer.extract_attention_params(director_output)

        c_update = 1./gamma * self.zoomer.zoom_out(glimpse, center_y, center_x, delta, sigma)

        return c_update # img_width * img_height

    def write_detailed(self, h_dec):
        glimpse = self.decoding_model.output(h_dec)
        director_output = self.director_model.output(h_dec)

        center_y, center_x, delta, sigma, gamma = self.zoomer.extract_attention_params(director_output)

        c_update = 1./gamma * self.zoomer.zoom_out(glimpse, center_y, center_x, delta, sigma)

        return c_update, center_y, center_x, delta