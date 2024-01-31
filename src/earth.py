from math import pi

import matplotlib.pyplot as plt
import numpy as np

# Erdumfang
U = 40000  # km
r = U / (2 * pi)

# Leuchtturm
h = 50 / 1000  # km

# Erdoberfläche
xs = np.linspace(0, 50)


def f_x(xs):
    """Works for scalar and vector. """
    return (r ** 2 - xs ** 2) ** (1 / 2) - r


ys = f_x(xs)


def f_x_dx(x):
    return -x / (r ** 2 - x ** 2) ** (1 / 2)


def tangente(p, xs):
    return f_x_dx(p) * (xs - p) + f_x(p)


# Berührpunkt der Tangente
p = (r ** 2 - (r ** 2 / (h + r)) ** 2) ** (1 / 2)

plt.plot(xs, ys)
plt.plot([0, 0], [0, h], color='red', linewidth=5)  # Leuchtturm
plt.plot(xs, tangente(p, xs))
plt.plot([p], [f_x(p)], marker="o")  # Berührpunkt

plt.xlim(0, 50)
plt.ylim(-0.3, h * 1.5)

plt.show()

# Calculate distance
dist = (p ** 2 + (h - f_x(p)) ** 2) ** (1 / 2)
print(f"Distance = {dist}")

