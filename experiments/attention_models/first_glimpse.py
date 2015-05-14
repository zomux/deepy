import time
import logging

from deepy import TrainerConfig
from deepy.dataset import MnistDataset, MiniBatches
from deepy.trainers import LearningRateAnnealer
from first_glimpse_model import get_network
from first_glimpse_trainer import FirstGlimpseTrainer
from deepy.utils import Timer

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("--model", default="/tmp/mnist_att_params2.gz")
    ap.add_argument("--method", default="MOMENTUM")
    ap.add_argument("--learning_rate", default=0.01)
    ap.add_argument("--variance", default=0.03)
    ap.add_argument("--disable_backprop", default=False)
    ap.add_argument("--disable_reinforce", default=False)
    ap.add_argument("--random_glimpse", default=False)
    args = ap.parse_args()

    mnist = MiniBatches((MnistDataset()), batch_size=1)

    model_path = args.model

    network = get_network(model_path, std=args.variance,
                          disable_reinforce=args.disable_reinforce, random_glimpse=args.random_glimpse)

    trainer_conf = TrainerConfig()
    trainer_conf.learning_rate = LearningRateAnnealer.learning_rate(args.learning_rate)
    trainer_conf.weight_l2 = 0.0001
    trainer_conf.hidden_l2 = 0.0001
    trainer_conf.method = args.method

    trainer = FirstGlimpseTrainer(network, network.layers[0], config=trainer_conf)

    annealer = LearningRateAnnealer(trainer, patience=5)

    timer = Timer()
    for _ in trainer.train(mnist.train_set(), mnist.valid_set(), mnist.test_set()):
        if annealer.invoke():
            break
    timer.end()

    network.save_params(model_path)

    timer.report()
