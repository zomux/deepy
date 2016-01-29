from . import NeuralLayer


class Combine(NeuralLayer):
    """
    Combine two variables.
    """

    def __init__(self, func, dim=0):
        """
        :type layer1: NeuralLayer
        :type layer2: NeuralLayer
        """
        super(Combine, self).__init__("combine")
        self.func = func
        if dim > 0:
            self.output_dim = dim

    def prepare(self):
        if self.output_dim == 0:
            self.output_dim = self.input_dim

    def compute_tensor(self, *tensors):
        return self.func(*tensors)

    def compute_test_tesnor(self, *tensors):
        return self.func(*tensors)
