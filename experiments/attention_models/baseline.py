import time
import logging

from deepy import TrainerConfig
from deepy.dataset import MnistDataset, MiniBatches
from experiments.attention_models.baseline_trainer import AttentionTrainer
from baseline_model import get_network

logging.basicConfig(level=logging.INFO)


mnist = MiniBatches((MnistDataset()), batch_size=1)

model_path = "/tmp/mnist_att_params2.gz"

network = get_network(model_path)

trainer_conf = TrainerConfig()
trainer_conf.learning_rate = 0.0012
trainer_conf.weight_l2 = 0.0001
trainer_conf.hidden_l2 = 0.0001
trainer_conf.monitor_frequency = trainer_conf.validation_frequency = trainer_conf.test_frequency = 1
trainer_conf.test_frequency = 10
trainer_conf.patience = 100

trainer = AttentionTrainer(network, network.layers[0], config=trainer_conf)

trainer_conf.report()

start_time = time.time()
c = 1
for k in list(trainer.train(mnist.train_set(), mnist.valid_set(), mnist.test_set())):
    if c > 10:
        break
    c += 1
print k
end_time = time.time()

network.save_params(model_path)

print "time:", ((end_time - start_time )/ 60), "mins"
