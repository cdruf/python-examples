# TODO


import numpy as np
from gekko import GEKKO

# Create model
n = 3
coeffs = np.random.rand(n)
m = GEKKO()
x_i = [m.Var(value=0, lb=0.0, ub=100.0, name=f"x_{i}") for i in range(n)]
obj_expr = m.sum([x_i[i] for i in range(n)])
m.Maximize(obj_expr)
m.Equation(m.sum(x_i) <= 50.0)

m2 = m.copy()

# Modify model 2
m.Equation(m.sum(x_i) <= 10.0)

# Solve both models
m.solve(disp=False)
m2.solve(disp=False)

# Print solution
for i in range(n):
    print(f"{x_i[i].NAME} = {x_i[i].VALUE[0]:.2f}")

