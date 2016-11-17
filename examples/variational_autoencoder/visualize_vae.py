#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from deepy import *
from train_vae import create_model

viz_path = os.path.join(os.path.dirname(__file__), "visualization.png")

if __name__ == '__main__':
    # Load the model
    model = create_model(load=True, sample=True)

    # Get first image in MNIST
    mnist = MnistDataset(for_autoencoder=True)
    first_img = mnist.train_set()[0]

    # Get the latent variable
    latent_variable = model.encode(first_img)[0]

    # Sample output images by varying the first latent variable
    deltas = np.linspace(-2, 2, 15)

    _, axmap = plt.subplots(len(deltas), len(deltas))
    for i, delta0 in enumerate(deltas):
        for j, delta1 in enumerate(deltas):
            new_variable = list(latent_variable)
            new_variable[0] += delta0
            new_variable[1] += delta1

            output_img = model.decode([np.array(new_variable, dtype=FLOATX)])
            output_img = output_img[0].reshape((28, 28))
            axmap[i, j].axis("off")
            axmap[i, j].matshow(output_img, cmap="gray")

    plt.savefig(viz_path, bbox_inches='tight', facecolor='black', dpi=60)
