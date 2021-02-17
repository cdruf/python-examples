# -*- coding: utf-8 -*-
"""

"""


from numba import jit
import numpy as np
from timeit import timeit

x = np.arange(10000).reshape(100, 100)

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def go_fast(a): # Function is compiled to machine code when called the first time
    trace = 0.0
    for i in range(a.shape[0]):   # Numba likes loops
        trace += np.tanh(a[i, i]) # Numba likes NumPy functions
    return a + trace              # Numba likes NumPy broadcasting

# compile it first
go_fast(x)

print(timeit('go_fast(x)', setup='from __main__ import x, go_fast', number=1000))


#%%
