# Experiment note of Variational autoencoders

## Commands

Train the VAE model on MNIST data.

```
python experiments/variational_autoencoder/train_vae.py
```

## About weight initialization

Sampling the weight parameters from normal distribution (sd=0.01) is crucial.

Sample from uniform distribution will give errors.

## About the training

It turns out that momentum training is not suitable and yields NaN in the gradients.

SGD works well but slow.

The original implementation uses modified AdaGrad.

In this example, AdaDelta is found to be better, Adam is found to be the best.

## Original implementation for VAE

[trainmnist.py](https://gist.github.com/zomux/41a228f57cfeb8de7994#file-trainmnist-py)

[variationalautoencoder.py](https://gist.github.com/zomux/4c5c077ff73f4213aef3#file-variationalautoencoder-py)