import numpy as np

from numba import jit, njit
from scipy.stats import variation


# It is not allowed to pass a callable with numba.

# dot is only possible for 1-D arrays, but we can reformulate it as follows

@jit(nopython=True)
def my_test_2(x, y):
    return (x * y).sum(axis=1)


x = np.array([[1, 2, 3, 1], [5, 4, 3, 5]])
y = np.array([1, 0, 1, 0])
result = my_test_2(x, y)
print(result)
print(x.dot(y))


# Clip works

@jit(nopython=True)
def my_test():
    return np.clip(np.array([1, 2, 3, 4, 5]), a_min=2, a_max=4)


print(my_test())

print('\n\n')


# STD works only without optional arguments

@njit
def my_test_std(x):
    ret = np.empty(x.shape[1])
    for col in range(x.shape[1]):
        xx = x[:, col]
        ret[col] = xx.std() / xx.mean()
    return ret


x = np.array([[1, 2, 3, 7, 8, 8, 8, 7, 1, 2, 3, 4, 3],
              [4, 5, 6, 1, 2, 3, 4, 5, 9, 9, 8, 2, 1],
              [1, 4, 3, 2, 9, 1, 2, 9, 4, 6, 6, 3, 5]])
print(my_test_std(x))
print(variation(x, axis=0))

import timeit

t1 = timeit.timeit(lambda: my_test_std(x), number=10000)
print(t1)
t2 = timeit.timeit(lambda: variation(x), number=10000)
print(t2)