import numpy as np
import scipy


def objective_function(xs: np.ndarray, args):
    return args[0] + np.sum(xs ** 2)


result = scipy.optimize.minimize(objective_function, x0=np.array([100, 100]), args=[100])

print(f"Optimal solution = ({', '.join(f'{i:.2f}' for i in result.x)})")
print(f"Optimal objective function value = {result.fun:.2f}")
