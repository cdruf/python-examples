import os

import numpy as np
import scipy
from matplotlib import pyplot as plt

_methods = ['Nelder-Mead', 'TNC', 'BFGS', 'Powell', 'L-BFGS-B', 'trust-constr', 'SLSQP']


def generate_data(n, min_weight=5, max_weight=10, min_lon=-117, max_lon=-77, min_lat=32, max_lat=41):
    alpha = np.random.rand(n)
    weights = min_weight + alpha * (max_weight - min_weight)
    alpha = np.random.rand(n)
    lons = min_lon + alpha * (max_lon - min_lon)
    alpha = np.random.rand(n)
    lats = min_lat + alpha * (max_lat - min_lat)
    return weights, lons, lats


def objective_function(y: np.ndarray, weights, lons, lats) -> float:
    return np.sum((np.sqrt((lons - y[0]) ** 2 + (lats - y[1]) ** 2)) * weights)


def optimize(weights, lons, lats):
    result = scipy.optimize.minimize(objective_function,
                                     x0=np.array([0.0, 0.0]),
                                     args=(weights, lons, lats),
                                     method=_methods[0])
    return result.x[0], result.x[1]


def visualize_simple(weights, lons, lats, x, y):
    plt.scatter(lons, lats, sizes=weights * 10)
    plt.scatter(x, y, sizes=[weights.sum() * 10])
    for idx, lon in enumerate(lons):
        lat = lats[idx]
        plt.plot([x, lon], [y, lat], color='grey')
    plt.show()


if __name__ == '__main__':
    print(os.getcwd())
    np.random.seed(3)
    ws, xs, ys = generate_data(20)
    x, y = optimize(ws, xs, ys)
    visualize_simple(ws, xs, ys, x, y)
