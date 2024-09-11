import os
import sys

import numpy as np
import scipy
from matplotlib import pyplot as plt

from optimization.weber_problem import generate_data
from util.math_helper import eq


def objective_function(y: np.ndarray, weights, lons, lats, n_locs) -> float:
    """
    y contains first the longitudes, then the latitudes.
    """
    yx = y[:n_locs]
    yy = y[n_locs:]
    ret = 0.0
    for i, lon in enumerate(lons):
        lat = lats[i]
        mini = sys.maxsize
        for j in range(n_locs):
            dist = np.sqrt((lon - yx[j]) ** 2 + (lat - yy[j]) ** 2)
            if dist < mini:
                mini = dist
        ret += mini * weights[i]
    return ret


def optimize(weights, lons, lats, n_locs):
    start_lons = np.repeat(lons.mean(), n_locs)
    start_lats = np.repeat(lats.mean(), n_locs)
    x0 = np.concatenate((start_lons, start_lats))
    result = scipy.optimize.minimize(objective_function,
                                     x0=x0,
                                     args=(weights, lons, lats, n_locs),
                                     method='Powell')
    return result.x[: n_locs], result.x[n_locs:]


def get_assignments(lons, lats, x_j, y_j, n_locs):
    assert n_locs == len(x_j) == len(y_j)
    ret = np.repeat(-1, len(lons))
    for i, lon in enumerate(lons):
        lat = lats[i]
        mini = sys.maxsize
        argmin = -1
        for j in range(n_locs):
            dist = np.sqrt((lon - x_j[j]) ** 2 + (lat - y_j[j]) ** 2)
            if dist < mini:
                mini = dist
                argmin = j
        ret[i] = argmin
    return ret


def visualize_simple(weights, lons, lats, x_j, y_j):
    plt.scatter(lons, lats, sizes=weights * 10)
    plt.scatter(x_j, y_j, sizes=[weights.sum() * 10])
    ass = get_assignments(lons, lats, x_j, y_j, n_locs)
    for idx, lon in enumerate(lons):
        lat = lats[idx]
        plt.plot([x_j[ass[idx]], lon], [y_j[ass[idx]], lat], color='grey')
    plt.show()


def check():
    n = 10
    ws, xs, ys = generate_data(n)
    assert eq(objective_function(y=np.concatenate([xs, ys]), weights=ws, lons=xs, lats=ys, n_locs=n), 0.0)


if __name__ == '__main__':
    print(os.getcwd())
    check()
    np.random.seed(7)
    ws, xs, ys = generate_data(50)
    n_locs = 3
    x_j, y_j = optimize(ws, xs, ys, n_locs)
    obj_value = objective_function(np.concatenate((x_j, y_j)), ws, xs, ys, n_locs)
    print(f"Objective function value = {obj_value:.2f}")
    visualize_simple(ws, xs, ys, x_j, y_j)
