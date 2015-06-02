# Tutorial 1: Your first *deepy* experiment

The first experiment with *deepy* is to build a simple multi-layer neural network with:

- two hidden layers with 256 neurons each
- ReLU activations
- Apply dropout with probability of 20% after ReLU
- Apply SGD with Momentum for model training

We use this network to classify MNIST digits, so the full architecture will be:

- 784-dimension input layer (28*28)
- 256-dimension fully-connected layer with ReLU activation
- Dropout layer
- 256-dimension fully-connected layer with ReLU activation
- Dropout layer
- 10-dimension fully-connected layer, no activation
- Softmax layer

Let's start.

## Checkout *deepy*

```bash
git clone https://github.com/uaca/deepy
```

## Setup your environment

Set your environment configurations properly or the experiments will be your nightmare.
Make sure you are not running Theano on one-core CPU.

If you are using a GPU machine, just execute this command:
```bash
source bin/gpu_env.sh
```

If you are using a multi-core CPU machine, execute this command:
```bash
source bin/cpu_env.sh
```

From now one, don't change directory.
 
Run this to create a directory and your first code.

```bash
mkdir my_experiments
touch my_experiments/tutorial1.py
```

If you are familiar with `vim`, your can edit the file with:
```bash
vi my_experiments/tutorial1.py
```

## Import classes you need

```python
# Set logging level so that you can see debug information of the training process.
import logging
logging.basicConfig(level=logging.INFO)

# Import classes
from deepy.dataset import MnistDataset, MiniBatches
from deepy.networks import NeuralClassifier
from deepy.layers import Dense, Softmax, Dropout
from deepy.trainers import MomentumTrainer, LearningRateAnnealer
```

## Define your model

```python
# Create a classifier, so it implies you will use cross-entropy as cost.
model = NeuralClassifier(input_dim=28*28) 
# Stack layers
model.stack(Dense(256, 'relu'),
            Dropout(0.2),
            Dense(256, 'relu'),
            Dropout(0.2),
            Dense(10, 'linear'),
            Softmax())
```


## Load training data

*deepy* has some pre-defined datasets like MNIST, so you just need to load them

```python
mnist = MiniBatches(MnistDataset(), batch_size=20)
```

## Define the training method

```python
trainer = MomentumTrainer(model)
# Note: you can specify options for the trainer
# For example, if you want to add L2 regularization
# trainer = MomentumTrainer(model, {"weight_l2": 0.0001})
```

For learning rate, if you want to modify it on the fly, you need define it a shared variable.
In *deepy* you can just do it like this:
```python
trainer = MomentumTrainer(model, {"learning_rate": LearningRateAnnealer.learning_rate(0.01)})
```

For a complete training option list, see this file:
[deepy/conf/trainer_config.py](https://github.com/uaca/deepy/blob/master/deepy/conf/trainer_config.py)

## Run the trainer

```python
# This will halve the learning rate if no lower valid cost is observed in 5 epochs
annealer = LearningRateAnnealer(trainer)

trainer.run(mnist, controllers=[annealer])
```

During the training process, you can just press "Ctrl + C" to quit. But do not press it more than once.

## Save your model

```python
model.save_params("tutorial1.gz")
```

To fill the model with saved parameters:
```python
model.load_params("tutorial1.gz")
```

## Finish editing your code, and run it

Run your code with:
```bash
python my_experiments/tutorial1.py
```

And after around 30 epochs, you can see this:
```
INFO:deepy.trainers.trainers:test    (iter=31) J=0.06 err=1.68
INFO:deepy.trainers.trainers:valid   (iter=31) J=0.07 err=1.69
```

If it does not go well, we have prepared a full code example for this tutorial.

Run it with:
```bash
python experiments/tutorials/tutorial1.py
```

The code list is here:
[Full code of this tutorial](https://github.com/uaca/deepy/blob/master/experiments/tutorials/tutorial1.py)