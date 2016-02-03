#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from deepy.networks import AutoEncoder
from deepy.layers import RNN, Dense
from deepy.trainers import SGDTrainer, LearningRateAnnealer

from util import get_data, VECTOR_SIZE, SEQUENCE_LENGTH

HIDDEN_SIZE = 50

model_path = os.path.join(os.path.dirname(__file__), "models", "rnn1.gz")

if __name__ == '__main__':
    model = AutoEncoder(rep_dim=10, input_dim=VECTOR_SIZE, input_tensor=3)
    model.stack_encoders(RNN(hidden_size=HIDDEN_SIZE, input_type="sequence", output_type="one"))
    model.stack_decoders(RNN(hidden_size=HIDDEN_SIZE, input_type="one", output_type="sequence", steps=SEQUENCE_LENGTH),
                         Dense(VECTOR_SIZE, 'softmax'))

    trainer = SGDTrainer(model)

    annealer = LearningRateAnnealer(trainer)

    trainer.run(get_data(), controllers=[annealer])

    model.save_params(model_path)

