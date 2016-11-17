import time
import logging
from argparse import ArgumentParser

from deepy import TrainerConfig
from deepy.dataset import MnistDataset, MiniBatches
from baseline_trainer import AttentionTrainer
from baseline_model import get_network
from deepy.utils import Timer

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("--model", default="/tmp/mnist_att_params2.gz")
    ap.add_argument("--method", default="momentum")
    ap.add_argument("--learning_rate", default=0.01)
    ap.add_argument("--variance", default=0.005)
    ap.add_argument("--disable_backprop", default=False)
    ap.add_argument("--disable_reinforce", default=False)
    ap.add_argument("--random_glimpse", default=False)
    args = ap.parse_args()

    mnist = MiniBatches((MnistDataset()), batch_size=1)

    model_path = args.model

    network = get_network(model_path, std=args.variance,
                          disable_reinforce=args.disable_reinforce, random_glimpse=args.random_glimpse)

    trainer_conf = TrainerConfig()
    trainer_conf.learning_rate = args.learning_rate
    trainer_conf.weight_l2 = 0.0001
    trainer_conf.hidden_l2 = 0.0001
    trainer_conf.method = args.method
    trainer_conf.disable_reinforce=args.disable_reinforce
    trainer_conf.disable_backprop=args.disable_backprop

    trainer = AttentionTrainer(network, network.layers[0], config=trainer_conf)

    trainer_conf.report()

    timer = Timer()
    for _ in list(trainer.train(mnist.train_set(), mnist.valid_set(), mnist.test_set())):
        pass
    timer.end()

    network.save_params(model_path)

    timer.report()
