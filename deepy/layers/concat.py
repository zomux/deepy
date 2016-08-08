import theano.tensor as T
from . import NeuralLayer


class Concatenate(NeuralLayer):
    """
    Concatenate two tensors.
    They should have identical dimensions except the last one.
    """

    def __init__(self, axis=-1):
        """
        :type layer1: NeuralLayer
        :type layer2: NeuralLayer
        """
        super(Concatenate, self).__init__("concate")
        self.axis = axis

    def prepare(self):
        self.output_dim = sum(self.input_dims)

    def compute_tensor(self, *xs):
        if self.axis == -1:
            axis = xs[0].ndim - 1
        else:
            axis = self.axis
        return T.concatenate(xs, axis=axis)