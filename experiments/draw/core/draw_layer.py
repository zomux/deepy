#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy import *
from attention_reader import AttentionReader
from attention_writer import AttentionWriter
from qsampler import Qsampler

ENCODER_HIDDEN_DIM = 256 # 256
DECODER_HIDDEN_DIM = 256 # 256
LATENT_VARIABLE_DIM = 100 # 100
READING_GLIMPSE_SIZE = 2
WRITING_GLIMPSE_SIZE = 5

READER_OUTPUT_DIM = 2 * READING_GLIMPSE_SIZE**2

LSTM_INIT = GaussianInitializer(deviation=0.01)
NORMAL_INIT = GaussianInitializer(deviation=0.01)


class DrawLayer(NeuralLayer):

    def __init__(self, img_width, img_height, attention_times):
        super(DrawLayer, self).__init__("draw")
        self.attention_times = attention_times
        self.img_width = img_width
        self.img_height = img_height

        self.reader = AttentionReader(DECODER_HIDDEN_DIM, img_width, img_height, READING_GLIMPSE_SIZE, init=NORMAL_INIT)
        self.encoder_lstm = LSTM(ENCODER_HIDDEN_DIM, inner_init=LSTM_INIT, outer_init=NORMAL_INIT, outer_activation='linear').connect(input_dim=READER_OUTPUT_DIM + DECODER_HIDDEN_DIM)
        self.sampler = Qsampler(LATENT_VARIABLE_DIM, init=NORMAL_INIT).connect(input_dim=ENCODER_HIDDEN_DIM)
        self.decoder_lstm = LSTM(DECODER_HIDDEN_DIM, inner_init=LSTM_INIT, outer_init=NORMAL_INIT).connect(input_dim=LATENT_VARIABLE_DIM)
        self.writer = AttentionWriter(DECODER_HIDDEN_DIM, img_width, img_height, WRITING_GLIMPSE_SIZE, init=NORMAL_INIT)

        self.register_inner_layers(self.reader.director_model, self.writer.director_model, self.writer.decoding_model)
        self.register_inner_layers(self.encoder_lstm, self.sampler, self.decoder_lstm)

    def _core_step(self, random_source, canvas, h_enc, c_enc, h_dec, c_dec, x):
        # Equation (3); Get the error of canvas
        x_hat = x - T.nnet.sigmoid(canvas)
        # Equation (4); Select an attention, read a glimpse
        r = self.reader.read(x, x_hat, h_dec)
        # Equation (5); Encode it with history
        encoder_inputs = self.encoder_lstm.produce_input_sequences(T.concatenate([r, h_dec], axis=1))
        encoder_inputs += [h_enc, c_enc]
        h_enc, c_enc = self.encoder_lstm.step(*encoder_inputs)
        # Equation (6); Sample a latent sample
        z_t, kl = self.sampler.sample(h_enc, random_source)
        # Equation (7); Decode latent sample
        decoder_inputs = self.decoder_lstm.produce_input_sequences(z_t)
        decoder_inputs += [h_dec, c_dec]
        h_dec, c_dec = self.decoder_lstm.step(*decoder_inputs)
        # Equation (8); Further decode it to glimpse, and restore glimpse to full-size image
        canvas = canvas + self.writer.write(h_dec)
        return canvas, h_enc, c_enc, h_dec, c_dec, z_t, kl


    def _get_outputs(self, x):
        batch_size = x.shape[0]

        # Sample from mean-zeros std.-one Gaussian
        random_sources = global_theano_rand.normal(
                    size=(self.attention_times, batch_size, LATENT_VARIABLE_DIM),
                    avg=0., std=1.)

        h_enc0 = T.alloc(np.cast[FLOATX](0.), batch_size, ENCODER_HIDDEN_DIM)
        h_dec0 = T.alloc(np.cast[FLOATX](0.), batch_size, DECODER_HIDDEN_DIM)

        [canvas, _, _, _, _, latent_var, kl], _ = theano.scan(self._core_step,
                                                              sequences=[random_sources],
                                                              outputs_info=[T.zeros_like(x), h_enc0, h_enc0, h_dec0, h_enc0, None, None],
                                                              non_sequences=[x],
                                                              n_steps=self.attention_times)

        x_drawn = T.nnet.sigmoid(canvas[-1,:,:])
        x_drawn.name = "_get_outputs"

        kl.name = "kl"

        return x_drawn, kl

    def output(self, x):
        """
        Directly output draw cost, Equation (12).
        """
        x_drawn, kl = self._get_outputs(x)
        kl_cost = kl.sum(axis=0).mean()
        # Clip to avoid NaN
        # x_drawn = T.clip(x_drawn, BIG_EPSILON, 1.0 - BIG_EPSILON)
        # x = T.clip(x, BIG_EPSILON, 1.0 - BIG_EPSILON)

        crossentropy = T.nnet.binary_crossentropy(x_drawn, x).sum(axis=1).mean()
        # self.register_monitors(("kl", kl_cost))
        cost =  crossentropy + kl_cost
        return cost

    def _decode_step(self, random_source, canvas, h_dec, c_dec):
        # Equation (13)
        z_t = self.sampler.sample_from_prior(random_source)
        # Equation (14)
        decoder_inputs = self.decoder_lstm.produce_input_sequences(z_t)
        decoder_inputs += [h_dec, c_dec]
        h_dec, c_dec = self.decoder_lstm.step(*decoder_inputs)
        # Equation (15)
        canvas = canvas + self.writer.write(h_dec)
        return canvas, h_dec, c_dec

    def sample(self, batch_size):
        """Sample from model.
        Returns
        -------
        samples : tensor3 (n_iter, n_samples, x_dim)
        """
        random_sources = global_theano_rand.normal(
                    size=(self.attention_times, batch_size, LATENT_VARIABLE_DIM),
                    avg=0., std=1.)

        h_dec0 = T.alloc(np.cast[FLOATX](0.), batch_size, DECODER_HIDDEN_DIM)
        canvas0 = T.zeros((batch_size, self.img_width * self.img_height))

        [canvas, _, _], _ = theano.scan(self._decode_step,
                                        sequences=[random_sources],
                                        outputs_info=[canvas0, h_dec0, h_dec0],
                                        n_steps=self.attention_times)
        return T.nnet.sigmoid(canvas)
