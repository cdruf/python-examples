import numpy as np
from scipy.optimize import approx_fprime


def my_function(xs: np.array) -> np.array:
    """R^2 --> R."""
    ret = np.sum(xs ** 0.5)
    return ret


jacobian = approx_fprime(xk=np.array([0.5, 0.5]), f=my_function)
print(jacobian)


def my_function_2(xs: np.array) -> np.array:
    """R^2 --> R^2."""
    ret = np.array([np.sum(xs ** 0.5), xs[0]])
    return ret


jacobian = approx_fprime(xk=np.array([0.5, 0.5]), f=my_function_2)
print(jacobian)
