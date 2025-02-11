import os

import matplotlib.pyplot as plt
import numpy as np


def seasonal(ts, period=365, amplitude=10):
    n = len(ts)
    return np.sin(np.arange(n) / period * 2 * np.pi) * amplitude


def cyclic(ts, cycle_length=2000, peak=90):
    a = -4 * peak / cycle_length ** 2
    b = -a * cycle_length
    x = np.arange(len(ts))
    ret = a * x ** 2 + b * x
    return ret


def irregular(n, sigma):
    return np.random.normal(0, sigma, n)


def get_level(ts, level):
    return np.repeat(level, len(ts))


def get_trend(ts, trend):
    return np.arange(len(ts)) * trend


def x_t(ts, level, trend, sigma):
    n = len(ts)
    lin = get_level(ts, level) + get_trend(ts, trend)
    ret = (lin + seasonal(ts) + cyclic(ts) + irregular(n, sigma))
    return ret


level = 100
trend = 0.05
ts = np.arange(365 * 4)
xs = x_t(ts, level, trend, 7)
fig, ax = plt.subplots()

ax.plot(ts, xs, label="Timeseries")
ax.plot(ts, get_level(ts, level), label="Level")
ax.plot(ts, get_trend(ts, trend), label="Trend")
ax.plot(ts, cyclic(ts), label="Cyclic")
ax.plot(ts, seasonal(ts), label="Seasonal")
ax.legend()
ax.set_ylabel("Demand")
ax.set_xlabel("Days")

print(os.getcwd())
plt.savefig("./my_timeseries_decomposed.png")
plt.show()
