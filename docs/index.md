# Welcome to deepy

*deepy* is a deep learning framework for designing models with complex architectures.

Many important components such as LSTM and Batch Normalization are implemented inside.

Although highly flexible, *deepy* maintains a clean high-level interface.

# Get started in 30 seconds

```
$ git clone https://github.com/zomux/deepy
$ cd deepy
$ pip install -r requirements.txt
$ source bin/cpu_env.sh
$ python experiments/mnist/mlp_dropout.py
```

# Learn more details in 3 minutes

## How to design your first simple neural network

First make a new python file for defining your first model.

```
$ mkdir my_experiments
$ touch my_experiments/first_model.py
```

Edit this file, perhaps with vim.
```
$ vim my_experiments/first_model.py
```

From now on, you are going to write python codes for defining a neural network.

First import everything from *deepy*.

```
from deepy.import_all import *
```

Suppose you want to design a multi-layer feed-forward network to classify MNIST numbers.

Then you have four questions to consider:

- What cost function to use
- What is the architecture of the network
- What optimization method to use
- Where is your dataset

With *deepy* you can implement a network easily and intuitively once you got the answers.

Here, we give a simple design of a feed-forward neural network.

```
model = NeuralClassifier(input_dim=28 * 28)

model.stack(
    Dense(256, 'relu'),
    Dense(256, 'relu'),
    Dense(10, 'linear'),
    Softmax()
)

trainer = AdaDeltaTrainer(model)

trainer.run(MiniBatches(MnistDataset(), batch_size=20))
```

Now you are done, simple run the following command to train your first model.

```
$ python my_experiments/first_model.py
```

You can also save your trained model by adding following code:

```
model.save_params("my_experiments/my_first_model.gz")
```

## Learn more

If you are willing to learn more about how to design a simple neural network, go to Tutorial 1.

# A brief overview of the classes in the framework

Here are the components of *deepy* framework, they are all designed in the spirit of simplicity.

![Overview of deepy](static/deepy.png)

