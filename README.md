deepy: Highly extensible deep learning framework based on Theano
===

   
[![Build](https://travis-ci.org/uaca/deepy.svg)](https://travis-ci.org/uaca/deepy)
[![Quality](https://img.shields.io/scrutinizer/g/uaca/deepy.svg)](https://scrutinizer-ci.com/g/uaca/deepy/?branch=master)
[![Requirements Status](https://requires.io/github/uaca/deepy/requirements.svg?branch=master)](https://requires.io/github/uaca/deepy/requirements/?branch=master)
[![Documentation Status](https://readthedocs.org/projects/deepy/badge/?version=latest)](http://deepy.readthedocs.org/en/latest/)
[![Coverage Status](https://coveralls.io/repos/uaca/deepy/badge.svg?branch=master)](https://coveralls.io/r/uaca/deepy?branch=master)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/uaca/deepy/blob/master/LICENSE)

### Dependencies

- numpy
- theano
- scipy for L-BFGS and CG optimization

Clean interface
===
```python
# MNIST Multi-layer model with dropout.
from deepy.dataset import MnistDataset, MiniBatches
from deepy.networks import NeuralClassifier
from deepy.layers import Dense, Softmax, Dropout
from deepy.trainers import MomentumTrainer, LearningRateAnnealer

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

Extensible model definition
===
```python

```

Extensible training procedure
===
```python

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
- Deep convolution
```
python experiments/mnist/deep_convolution.py
```
- Recurrent visual attention model
   - [Result visualization](http://raphael.uaca.com/experiments/recurrent_visual_attention/Plot%20attentions.html)
```
python experiments/attention_models/baseline.py
```

### Language model

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

Other features
===

- Auto gradient correction

```
Don't ask for a document, the code itself is so easy to understand,
that it does not require a redundant document to explain.
```
Raphael Shu, 2015