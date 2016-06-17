deepy: A highly extensible deep learning framework based on Theano
===
[![Build](https://travis-ci.org/zomux/deepy.svg)](https://travis-ci.org/zomux/deepy)
[![Quality](https://img.shields.io/scrutinizer/g/zomux/deepy.svg)](https://scrutinizer-ci.com/g/zomux/deepy/?branch=master)
[![PyPI version](https://badge.fury.io/py/deepy.svg)](https://badge.fury.io/py/deepy)
[![Requirements Status](https://requires.io/github/zomux/deepy/requirements.svg?branch=master)](https://requires.io/github/zomux/deepy/requirements/?branch=master)
[![Documentation Status](https://readthedocs.org/projects/deepy/badge/?version=latest)](http://deepy.readthedocs.org/en/latest/)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/zomux/deepy/blob/master/LICENSE)

*deepy* is a deep learning framework for designing models with complex architectures.

Many important components such as LSTM and Batch Normalization are implemented inside.

Although highly flexible, *deepy* maintains a clean high-level interface.

From deepy 0.2.0, you can easily design very complex computational graphs such as Neural Turing Machines.

Example codes will be added shortly.

## Recent updates

deepy now supports training on multiple GPUs, see the following example for training neural machine translation models.

https://github.com/zomux/neuralmt

## Dependencies

- Python 2.7 (Better on Linux)
- numpy
- theano
- scipy for L-BFGS and CG optimization

### Tutorials (Work in progress)

[http://deepy.readthedocs.org/en/latest/](http://deepy.readthedocs.org/en/latest/)

Clean interface
===
```python
# A multi-layer model with dropout for MNIST task.
from deepy import *

model = NeuralClassifier(input_dim=28*28)
model.stack(Dense(256, 'relu'),
            Dropout(0.2),
            Dense(256, 'relu'),
            Dropout(0.2),
            Dense(10, 'linear'),
            Softmax())

trainer = MomentumTrainer(model)

annealer = LearningRateAnnealer(trainer)

mnist = MiniBatches(MnistDataset(), batch_size=20)

trainer.run(mnist, controllers=[annealer])
```

Examples
===

### Enviroment setting

- CPU
```
source bin/cpu_env.sh
```
- GPU
```
source bin/gpu_env.sh
```

### MNIST Handwriting task

- Simple MLP
```
python experiments/mnist/mlp.py
```
- MLP with dropout
```
python experiments/mnist/mlp_dropout.py
```
- MLP with PReLU and dropout
```
python experiments/mnist/mlp_prelu_dropout.py
```
- Maxout network
```
python experiments/mnist/mlp_maxout.py
```
- Deep convolution
```
python experiments/mnist/deep_convolution.py
```
- Elastic distortion
```
python experiments/mnist/mlp_elastic_distortion.py
```
- Recurrent visual attention model
   - [Result visualization](http://raphael.uaca.com/experiments/recurrent_visual_attention/Plot%20attentions.html)
```
python experiments/attention_models/baseline.py
```

### Variational auto-encoders

- Train a model
```
python experiments/variational_autoencoder/train_vae.py
```

- Visualization the output when varying the 2-dimension latent variable
```
python experiments/variational_autoencoder/visualize_vae.py
```

- Result of visualization

![](https://raw.githubusercontent.com/uaca/deepy/master/experiments/variational_autoencoder/visualization.png)

### Language model

#### Penn Treebank benchmark

- Baseline RNNLM (Full-output layer)
```
python experiments/lm/baseline_rnnlm.py
```
- Class-based RNNLM
```
python experiments/lm/class_based_rnnlm.py
```
- LSTM based LM (Full-output layer)
```
python experiments/lm/lstm_rnnlm.py
```

#### Char-based language models

- Char-based LM with LSTM
```
python experiments/lm/char_lstm.py
```
- Char-based LM with Deep RNN
```
python experiments/lm/char_rnn.py
```

### Deep Q learning

- Start server
```
pip install Flask-SocketIO
python experiments/deep_qlearning/server.py
```
- Open this address in browser
```
http://localhost:5003
```

### Auto encoders

- Recurrent NN based auto-encoder
```
python experiments/auto_encoders/rnn_auto_encoder.py
```
- Recursive auto-encoder
```
python experiments/auto_encoders/recursive_auto_encoder.py
```

### Train with CG and L-BFGS

- CG
```
python experiments/scipy_training/mnist_cg.py
```
- L-BFGS
```
python experiments/scipy_training/mnist_lbfgs.py
```
Other experiments
===

### DRAW

See https://github.com/uaca/deepy-draw

```
# Train the model
python mnist_training.py
# Create animation
python animation.py experiments/draw/mnist1.gz
```

![](https://github.com/uaca/deepy-draw/raw/master/plots/mnist-animation.gif)

### Highway networks

- http://arxiv.org/abs/1505.00387
```
python experiments/highway_networks/mnist_baseline.py
python experiments/highway_networks/mnist_highway.py
```

### Effect of different initialization schemes

```
python experiments/initialization_schemes/gaussian.py
python experiments/initialization_schemes/uniform.py
python experiments/initialization_schemes/xavier_glorot.py
python experiments/initialization_schemes/kaiming_he.py
```


---

Sorry for that deepy is not well documented currently, but the framework is designed in the spirit of simplicity and readability.
This will be improved if someone requires.

**Raphael Shu, 2016**
