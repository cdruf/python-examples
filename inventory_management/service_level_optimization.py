"""
Optimize the service level.

Assumptions:

+ normally distributed demand,


"""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from scipy.stats import norm

from inventory_management.inventory_data import Data
from inventory_management.loss_functions import units_short


def safety_stock(z, sig):
    """z = safety factor. """
    return z * sig


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

    def get_optimal_alpha(self):
        if self.data.backlog_cost < 0.000001:
            return 0.0
        return 1.0 - self.data.holding_cost * self.R / self.data.backlog_cost


def create_static_matplot():
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
    plt.plot(alphas, np.repeat(data.fixed_replenishment_cost / rs.R, len(alphas)), label='Transaction')
    plt.ylim(0, 800)
    plt.legend()
    plt.show()


def create_dynamic_plotly_chart():
    min_y = 0
    max_y = 800
    bs = np.arange(0, 101, step=10)
    min_alpha = 0.001
    max_alpha = 0.990
    alphas = np.linspace(min_alpha, max_alpha, 600)
    zs = norm.ppf(alphas)

    instances = {b: RsPolicy(Data(demand_mu=100.0, demand_sig=25.0, fixed_replenishment_cost=200.0, holding_cost=2.0,
                                  lead_time=1.0,
                                  backlog_cost=b), R=1.0, S=300)
                 for b in bs}

    fig = go.Figure(layout_yaxis_range=[min_y, max_y], layout_xaxis_range=[min_alpha, max_alpha])

    idx_map = {}  # b -> index
    idx = 0
    b_map = {}  # index -> b

    # Add traces, one for each slider step
    for b in bs:
        idx_map[b] = idx
        b_map[idx] = b
        idx += 1

        instance: RsPolicy = instances[b]

        costs_per_period = np.array([instance.cost_per_period(zs[idx]) for idx, a in enumerate(alphas)])

        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#00CED1", width=6),
                # name=f"Cost per backorder = {b}",
                name="Total costs",
                x=alphas,
                y=costs_per_period))

    # Add traces for optimal service level
    for b in bs:
        alpha_opt = instances[b].get_optimal_alpha()
        # z_opt = norm.ppf(alpha_opt)
        # print(z_opt)
        fig.add_trace(
            go.Scatter(
                visible=False,
                mode='lines',
                line=dict(color="#0A1335", width=1),
                name=f"Optimum",
                x=[alpha_opt, alpha_opt],
                y=[min_y, max_y]))

    # Make 1st trace visible
    idx_of_initial_b = idx_map[50]
    fig.data[idx_of_initial_b].visible = True
    fig.data[len(bs) + idx_of_initial_b].visible = True

    # Create and add fixed cost slider
    backorder_cost_slider = []
    for idx, b in enumerate(bs):
        visible_mask = [False] * len(fig.data)
        visible_mask[idx] = True  # Make the curve visible
        visible_mask[len(bs) + idx] = True  # Make the optimum visibl
        step_config = dict(
            method="update",
            args=[{"visible": visible_mask},
                  # {"title": f"Backorder cost = {b}"}
                  ],
            label=f"{b}",
        )
        backorder_cost_slider.append(step_config)

    sliders = [dict(
        active=idx_of_initial_b,
        currentvalue={"prefix": "Backorder cost: "},
        pad={"t": 50},
        steps=backorder_cost_slider
    )]

    tickvals = [round(i / 10, 1) for i in range(11)]
    ticktext = [f'{val * 100:.0f}%' for val in tickvals]
    fig.update_layout(
        sliders=sliders,
        xaxis_title="Service level",
        yaxis_title="Total costs",
        showlegend=True,
        xaxis=dict(
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext
        )
    )

    fig.show()
    fig.write_html("../data/service_level_plotly.html", full_html=False, include_plotlyjs=False)
    """
    Diese Zeile vor dem exportierten code hinzufügen:
    <script src="https://cdn.plot.ly/plotly-2.25.2.min.js" charset="utf-8"></script>
    """


if __name__ == '__main__':
    print('What do you want to do?')
    print('1 - static matplot')
    print('2 - dynamic plotly')
    option = int(input('Enter number: \n'))
    if option == 1:
        create_static_matplot()
    elif option == 2:
        create_dynamic_plotly_chart()
    print('Adios ... pidele trabajo!')
