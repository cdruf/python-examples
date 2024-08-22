import numpy as np
from scipy.optimize import minimize, Bounds


def my_non_linear_function(x: np.array):
    assert x.shape[0] == 2
    return x[0] * x[1]


def obj(x):
    return my_non_linear_function(x)


def c1(x):
    """ sum(abs(x)) >= 5 """
    return np.abs(x)._sum() - 5


sol = minimize(
    fun=obj,
    x0=np.array([0] * 2),
    method='SLSQP',
    bounds=Bounds(0, 100000),
    constraints=[{'type': 'ineq', 'fun': c1}],  # inequality means that it is to be non-negative
    options={'maxiter': 1000})

print(sol)
