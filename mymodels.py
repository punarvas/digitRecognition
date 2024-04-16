import neuralnetwork as nn
import numpy as np

# Classification error
metrics = nn.NNetMetric(f=nn.nnet_error_rate)
np.random.seed(42)


def model_1(n: int, K: int, learning_rate: float = 0, iterations: int = 50):
    # n = Number of input units, K = number of output units
    model = nn.NNet(nunits=[n, K])
    optimizer = nn.NNetGDOptimizer(metric=metrics, max_iters=iterations, learn_rate=learning_rate)
    return model, optimizer


def model_2(n: int, K: int, hidden_units: int, depth: int, learning_rate: float = 0, iterations: int = 50):
    # hidden_units = number of hidden units, depth = depth of the model
    n_units = nn.make_nunits(n, K, depth, hidden_units)
    model = nn.NNet(nunits=n_units)
    optimizer = nn.NNetGDOptimizer(metric=metrics, max_iters=iterations, learn_rate=learning_rate)
    return model, optimizer


def model_3(n: int, K: int, hidden_units: int, learning_rate: float = 0, iterations: int = 50):
    n_units = [n]
    m = hidden_units
    for i in range(2):
        n_units.append(hidden_units)
        m = int(m / 4)
    n_units.append(K)
    model = nn.NNet(nunits=n_units)
    optimizer = nn.NNetGDOptimizer(metric=metrics, max_iters=iterations, learn_rate=learning_rate)
    return model, optimizer
