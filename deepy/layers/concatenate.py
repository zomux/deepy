import theano.tensor as T
from . import NeuralLayer


class Concatenate(NeuralLayer):
    """
    Concatenate two tensors.
    They should have identical dimensions except the last one.
    """

    def __init__(self, layer1, layer2, axis=2):
        """
        :type layer1: NeuralLayer
        :type layer2: NeuralLayer
        """
        super(Concatenate, self).__init__("concate")
        self.layer1 = layer1
        self.layer2 = layer2
        self.axis = axis
        if axis < 0:
            raise Exception("There are some bugs in T.concatenate, that axis can not lower than 0")

    def setup(self):
        self.layer1.connect(self.input_dim)
        self.layer2.connect(self.input_dim)
        self.output_dim = self.layer1.output_dim + self.layer2.output_dim
        self.register_inner_layers(self.layer1, self.layer2)

    def output(self, x):
        tensor1 = self.layer1.output(x)
        tensor2 = self.layer2.output(x)
        return T.concatenate([tensor1, tensor2], axis=self.axis)

    def test_output(self, x):
        tensor1 = self.layer1.test_output(x)
        tensor2 = self.layer2.test_output(x)
        return T.concatenate([tensor1, tensor2], axis=self.axis)
