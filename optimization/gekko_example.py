from time import time

import numpy as np
from gekko import GEKKO

# Create example data
n = 10
const = 5.0
coeffs = np.random.rand(n)
coeffs = dict(zip(range(n), coeffs.tolist()))

# Create model
m = GEKKO()

# Add variables
x_i = {i: m.Var(value=0, lb=0.0, ub=100.0, name=f"x_{i}") for i in range(n)}
y_i = {i: m.Intermediate(const + coeffs[i] * x_i[i], name=f"y_{i}") for i in range(n)}

# Add useless intermediates
z_i = {i: m.Intermediate(y_i[i], name=f"z_{i}") for i in range(n)}

# Add objective
obj_expr = m.sum([y_i[i] for i in range(n)])
m.Maximize(obj_expr)

# Add constraints
m.Equation(m.sum([x for x in x_i.values()]) <= 50.0)

# for i in range(n):
#     m.Equation([const + coeffs[i] * x_i[i] == y_i[i]])

# Solve model
start = time()
m.options.IMODE = 3  # steady state optimization
m.options.SOLVER = 1  # APOPT solver
m.solve(disp=False)
duration_sec = time() - start
print(f"Solved in {duration_sec} seconds.")

# Print solution
for i in range(n):
    print(f"{x_i[i].NAME} = {x_i[i].VALUE[0]:.2f}")
    print(f"{y_i[i].NAME} = {y_i[i].VALUE[0]:.2f}")
