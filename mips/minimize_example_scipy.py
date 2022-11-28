import numpy as np
from scipy.optimize import minimize, Bounds


def obj(x):
    return x.sum()


def c1(x):
    return x.sum() - 5


sol = minimize(
    fun=obj,
    x0=np.array([0] * 2),
    method='SLSQP',
    bounds=Bounds(0, 100000),
    constraints=[{'type': 'ineq', 'fun': c1}],  # inequality means that it is to be non-negative
    options={'maxiter': 1000})

print(sol)
