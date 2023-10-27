import matplotlib.pyplot as plt
import numpy as np

xs = np.linspace(-10, 10, 1000)
# ys = (np.sin(xs) + (1.1 * xs) - 4)
# ys = 0.1 * xs ** 2 - 5
ys = np.sin(xs)

plt.plot(xs, ys)
plt.ylim(-1.2, 1.2)
plt.xlim(-10, 10)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x) = y')
plt.show()
