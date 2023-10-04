import numpy as np
import plotly.graph_objects as go

from inventory_management.economic_order_quantity import EOQ
from inventory_management.inventory_data import Data
from inventory_management.safety_stocks import get_reorder_point_and_safety_stock_level

EPS = 1e-4


def create_dynamic_plotly_chart():
    min_y = 0
    demand_mu = 12000.0
    sigmas = np.arange(500, 6001, step=500)
    cvs = [sigma / demand_mu for sigma in sigmas]
    min_x = 0
    max_x = 365  # on year
    ts = np.linspace(min_x, max_x, 600)  # time points

    instances = {sigma: Data(demand_mu=demand_mu, demand_sig=sigma, fixed_replenishment_cost=50.0, holding_cost=0.1,
                             lead_time=1.0, backlog_cost=10.0)
                 for sigma in sigmas}

    # We calculate the EQO only once because we only vary the safety stock
    eoq = EOQ.from_data_object(data=next(iter(instances.values())),
                               r=0.1)  # Can use any instance from the list as only the STD differs
    eoq.compute()

    target_fill_rate = 0.95

    # Calculate safety stocks
    idx_map = {}  # sigma -> index
    idx = 0
    sigma_map = {}  # index -> sigma
    safety_stock_sigma = {}  # sigma -> SS
    for sigma in sigmas:
        idx_map[sigma] = idx
        sigma_map[idx] = sigma
        idx += 1
        instance: Data = instances[sigma]

        # Calculate the mean demand in the risk period and the STD
        mean = instance.demand_mu * eoq.T
        std = np.sqrt(eoq.T * instance.demand_sig ** 2)
        rop, safety_stock = get_reorder_point_and_safety_stock_level(
            mu=mean, std=std, target_fill_rate=target_fill_rate)
        print(safety_stock)

        safety_stock_sigma[sigma] = safety_stock

    # Calculate max y for figure
    max_y = eoq.Q + max(safety_stock_sigma.values())

    fig = go.Figure(layout_yaxis_range=[min_y, max_y], layout_xaxis_range=[min_x, max_x])

    # Add traces for the safety stock line, one for each slider step
    for idx, sigma in enumerate(sigmas):
        cv = cvs[idx]
        safety_stocks = np.repeat(safety_stock_sigma[sigma], len(ts))

        fig.add_trace(
            go.Scatter(
                visible=False,
                mode='lines',
                line=dict(color="#0A1335", width=1),
                # name=f"Forecast error = {sigma}",
                name="Safety stock",
                x=ts,
                y=safety_stocks))

    # Add traces for the inventory level, one for each slider step
    for idx, sigma in enumerate(sigmas):
        cv = cvs[idx]
        xs, ys = eoq.get_xs_and_ys()
        ys = np.array(ys)
        ss = safety_stock_sigma[sigma]
        ys += ss
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#00CED1", width=6),
                # name=f"Forecast error = {sigma}",
                name="Total stock",
                x=xs,
                y=ys))

    # Make 1st trace visible
    idx_of_initial_sigma = len(sigmas) // 2
    fig.data[idx_of_initial_sigma].visible = True  # Make safety stock line visible
    fig.data[len(sigmas) + idx_of_initial_sigma].visible = True  # Make inventory line visible

    # Create and add fixed cost slider
    forecast_error_slider = []
    for idx, sigma in enumerate(sigmas):
        cv = cvs[idx]
        visible_mask = [False] * len(fig.data)
        visible_mask[idx] = True  # Make the safety stock line visible
        visible_mask[len(sigmas) + idx] = True  # Make the inventory line visible
        step_config = dict(
            method="update",
            args=[{"visible": visible_mask},
                  # {"title": f"Forecast error = {sigma}"}
                  ],
            label=f"{cv * 100:.0f}%",
        )
        forecast_error_slider.append(step_config)

    sliders = [dict(
        active=idx_of_initial_sigma,
        currentvalue={"prefix": "Forecast error: "},
        pad={"t": 50},
        steps=forecast_error_slider
    )]

    fig.update_layout(
        sliders=sliders,
        xaxis_title="Days of year",
        yaxis_title="On-hand inventory",
        showlegend=True,
    )

    fig.show()
    fig.write_html("../data/safety_stock.html", full_html=False, include_plotlyjs=False)
    """
    Diese Zeile vor dem exportierten code hinzufügen:
    <script src="https://cdn.plot.ly/plotly-2.25.2.min.js" charset="utf-8"></script>
    """


if __name__ == '__main__':
    print('Hello!')
    create_dynamic_plotly_chart()
    print('Adios ... pidele trabajo!')
