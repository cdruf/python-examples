import numpy as np
from scipy.optimize import linprog

c = -np.array([3, 2])
b_ub = np.array([100, 80, 40])
A_ub = np.array([[2, 1],
                 [1, 1],
                 [1, 0]])

res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, method='highs')

print(res)

