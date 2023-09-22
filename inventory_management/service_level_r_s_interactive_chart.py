"""
Optimize the service level.

Assumptions:

+ (R, S) policy,
+ normally distributed demand,

"""
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm


@dataclass
class Data:
    demand_mu: float
    demand_sig: float
    fixed_replenishment_cost: float
    backlog_cost: float
    holding_cost: float
    lead_time: float


def safety_stock(z, sig):
    """z = safety factor. """
    return z * sig


def loss_function_standard_normal(x: float) -> float:
    """
    :param x: Safety factor.
    """
    return norm.pdf(x) - x * (1.0 - norm.cdf(x))


def loss_function_normal(x: float, mu: float, sigma: float) -> float:
    if sigma == 0:
        return max(mu - x, 0.0)
    return sigma * loss_function_standard_normal((x - mu) / sigma)


def units_short(demand_sig, z):
    return demand_sig * loss_function_standard_normal(z)


@dataclass
class RsPolicy:
    data: Data
    R: float
    S: float

    def cycle_stock(self):
        return self.data.demand_mu * self.R / 2.0

    def sigma_x(self):
        """Variance over risk period."""
        return np.sqrt(self.R + self.data.lead_time) * self.data.demand_sig

    def exp_avg_on_hand_inventory(self, z):
        """
        S = demand during lead time + demand during review period + SS.
        demand during review period = demand_mu x review period.
        Inventory level varies between S - d_L = d_R + SS and SS.
        Average level = d_R / 2 + SS.
        """
        return self.cycle_stock() + safety_stock(z, self.sigma_x())

    def holding_cost_per_period(self, z):
        return self.data.holding_cost * self.exp_avg_on_hand_inventory(z)

    def backorder_cost_per_period(self, z):
        return self.data.backlog_cost * units_short(self.sigma_x(), z)

    def cost_per_review_period(self, z):
        holding = self.holding_cost_per_period(z) * self.R
        return holding + self.data.fixed_replenishment_cost + self.backorder_cost_per_period(z)

    def cost_per_period(self, z):
        return self.cost_per_review_period(z) / self.R


if __name__ == '__main__':
    data = Data(demand_mu=100.0,
                demand_sig=25.0,
                fixed_replenishment_cost=200.0,
                backlog_cost=50.0,
                holding_cost=2.0,
                lead_time=1.0)
    rs = RsPolicy(data, R=1.0, S=300)
    alphas = np.linspace(0.6, 0.99, 1000)
    zs = norm.ppf(alphas)
    plt.plot(alphas, np.array([rs.cost_per_period(zs[idx]) for idx, a in enumerate(alphas)]), label='Total')
    plt.plot(alphas, np.array([rs.holding_cost_per_period(zs[idx]) for idx, a in enumerate(alphas)]),
             label='Holding')
    plt.plot(alphas, np.array([rs.backorder_cost_per_period(zs[idx]) for idx, a in enumerate(alphas)]),
             label='Backorder')
    plt.plot(alphas, np.repeat(data.fixed_replenishment_cost, len(alphas)), label='Transaction')
    plt.ylim(0, 800)
    plt.legend()
    plt.show()
