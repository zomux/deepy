from deepy.core.env import FLOATX
import theano.tensor as T
from . import NeuralLayer

class Dense(NeuralLayer):
    """
    Fully connected layer.
    """

    def __init__(self, size, activation='linear', init=None, disable_bias=False, random_bias=False, wnorm=False):
        super(Dense, self).__init__("dense")
        self.activation = activation
        self.output_dim = size
        self.disable_bias = disable_bias
        self.random_bias = random_bias
        self.initializer = init
        self._wnorm = wnorm

    def prepare(self):
        self._setup_params()
        self._setup_functions()

    def compute_tensor(self, x):
        if self._wnorm:
            fix = (self.g/T.sqrt(T.sum(T.square(self.W),axis=0)))[None, :]
            W = self.W *fix
        else:
            W = self.W
        return self._activation(T.dot(x, W) + self.B)

    def _setup_functions(self):
        from deepy.tensor.activations import get_activation
        self._activation = get_activation(self.activation)

    def _setup_params(self):
        self.W = self.create_weight(self.input_dim, self.output_dim, initializer=self.initializer)
        self.register_parameters(self.W)
        if self._wnorm:
            self.g = self.create_weight(shape=(self.output_dim,), label="g")
            self.register_parameters(self.g)
        if self.disable_bias:
            self.B = T.constant(0, dtype=FLOATX)
        elif self.random_bias:
            self.B = self.create_weight(initializer=self.initializer,
                                        shape=(self.output_dim, ))
            self.register_parameters(self.B)
        else:
            self.B = self.create_bias(self.output_dim, "B")
            self.register_parameters(self.B)
