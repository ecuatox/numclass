import numpy as np

def sigmoid(x, deriv=False):
    if deriv:
        return x * (1-x)
    return 1 / (1+np.exp(-x))

class Net:
    def __init__(self, layer_sizes, activation=sigmoid):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.weights = np.array([2 * np.random.rand(layer_sizes[i], layer_sizes[i+1]) - 1 for i in range(len(layer_sizes)-1)])
        self.layers = None

    def __len__(self):
        return len(self.layer_sizes)

    def save(self, filename):
        np.save(filename, self.weights)

    @staticmethod
    def load(filename):
        weights = np.load(filename)
        layer_sizes = [len(a) for a in weights]
        layer_sizes.append(len(weights[-1][0]))
        net = Net(layer_sizes)
        net.weights = weights
        return net

    def shuffle(self):
        self.weights = np.array([2 * np.random.rand(self.layer_sizes[i], self.layer_sizes[i + 1]) - 1 for i in range(len(self.layer_sizes) - 1)])

    def run(self, x):
        self.layers = [None for _ in self.layer_sizes]
        self.layers[0] = np.array(x)
        for i in range(len(self)-1):
            x = self.activation(np.dot(x, self.weights[i]))
            self.layers[i+1] = x
        return x

    def mean_error(self, target):
        return np.mean(np.abs(target-self.layers[-1]))

    def train(self, x, target):
        result = self.run(x)
        errors = [None for _ in range(len(self))]
        deltas = [None for _ in range(len(self))]
        errors[-1] = target - result

        for i in range(len(self)-1, 0, -1):
            deltas[i] = errors[i] * self.activation(self.layers[i], deriv=True)
            errors[i-1] = deltas[i].dot(self.weights[i-1].T)

        for i in range(len(self)-2, -1, -1):
            self.weights[i] += self.layers[i].T.dot(deltas[i+1]) * 0.03
