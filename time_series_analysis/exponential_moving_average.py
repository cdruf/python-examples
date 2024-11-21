import numpy as np
from matplotlib import pyplot as plt

xs = np.arange(16)
ys = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

alpha = 0.4

ema_1 = np.zeros(16)
ema_1[0] = alpha * 0 + (1 - alpha) * ys[0]  # seed is assumed to be zero
for i in range(1, 16):
    ema_1[i] = alpha * ys[i] + (1 - alpha) * ema_1[i - 1]

ema_2 = np.zeros(16)
ema_2[0] = ys[0]  # seed is assumed to be the first value
for i in range(1, 16):
    ema_2[i] = alpha * ys[i] + (1 - alpha) * ema_2[i - 1]

ema_3 = np.zeros(16)
ema_3[0] = ys[0]
denom = 1
denom_last_term = 1
denom_alt = 1 - alpha
for i in range(1, 16):
    e = ys[i] + (1 - alpha) * ema_3[i - 1] * denom
    denom_last_term *= (1 - alpha)
    denom += denom_last_term
    ema_3[i] = e / denom

    denom_alt *= (1 - alpha)
    tmp = (1 - denom_alt) / alpha
    print(denom, tmp)

ema_4 = np.zeros(16)
ema_4[0] = ys[0]
denom = 1 - alpha
for i in range(1, 16):
    e = alpha * ys[i] + (1 - alpha) * ema_4[i - 1] * (1 - denom)
    denom *= (1 - alpha)
    ema_4[i] = e / denom

plt.scatter(xs, ys, s=60)
plt.plot(xs, ema_1, marker="*", linestyle=":", label="0")
plt.plot(xs, ema_2, marker="*", linestyle=":", label="1st")
plt.plot(xs, ema_3, marker="*", linestyle=":", label="correct-1")
# plt.plot(xs, ema_4, marker="*", linestyle=":", label="correct-2")
plt.legend(loc="upper right")
plt.show()
