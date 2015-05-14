from deepy.utils import build_activation, FLOATX
import theano.tensor as T
from . import NeuralLayer

class Dense(NeuralLayer):
    """
    Fully connected layer.
    """

    def __init__(self, size, activation='linear', init=None, disable_bias=False):
        super(Dense, self).__init__("dense")
        self.activation = activation
        self.output_dim = size
        self.disable_bias = disable_bias
        self.initializer = init

    def setup(self):
        self._setup_params()
        self._setup_functions()

    def output(self, x):
        return self._activation(T.dot(x, self.W) + self.B)

    def _setup_functions(self):
        self._activation = build_activation(self.activation)

    def _setup_params(self):
        self.W = self.create_weight(self.input_dim, self.output_dim, self.name, initializer=self.initializer)
        self.register_parameters(self.W)
        if self.disable_bias:
            self.B = T.constant(0, dtype=FLOATX)
        else:
            self.B = self.create_bias(self.output_dim, self.name)
            self.register_parameters(self.B)
