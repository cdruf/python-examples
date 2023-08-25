import math
from dataclasses import dataclass, field

from matplotlib import pyplot as plt


@dataclass
class EOQ:
    D: int
    A: float
    v: float
    r: float

    T: float = field(init=False)
    Q: float = field(init=False)

    def compute(self):
        self.Q = math.sqrt(2 * self.A * self.D / (self.v * self.r))
        self.T = self.Q / self.D

    def print(self):
        print(f"EOQ = {self.Q:.1f}, order interval = {self.T:.1f}")

    def get_xs_and_ys(self):
        order_cycles_per_year = math.ceil(1.0 / self.T)
        xs = []
        ys = []
        days = 365 * self.T
        for i in range(order_cycles_per_year):
            xs.append(i * days)
            ys.append(self.Q)
            xs.append((i + 1) * days)
            ys.append(0)
        return xs, ys

    def plot(self):
        xs, ys = self.get_xs_and_ys()
        plt.plot(xs, ys)
        plt.show()
