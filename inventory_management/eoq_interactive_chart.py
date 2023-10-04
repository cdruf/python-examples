import numpy as np
import plotly.graph_objects as go

from inventory_management.economic_order_quantity import EOQ

A_min = 1
A_max = 100

eoqs = {a: EOQ(D=12000, A=a, v=1.0, r=0.1) for a in range(A_min, A_max + 1)}
q_max = 0
for eoq in eoqs.values():
    eoq.compute()
    if eoq.Q > q_max:
        q_max = eoq.Q

# Create figure
fig = go.Figure(layout_yaxis_range=[0, q_max], layout_xaxis_range=[0, 365])

idx_map = {}
idx = 0
A_map = {}

# Add traces, one for each slider step
for fixed_cost in np.arange(A_min, A_max + 1, 1):
    idx_map[fixed_cost] = idx
    A_map[idx] = fixed_cost
    idx += 1
    xs, ys = eoqs[fixed_cost].get_xs_and_ys()
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color="#00CED1", width=6),
            #name=f"A = {fixed_cost}",
            name="Cycle stock",
            x=xs,
            y=ys))

# Make 1st trace visible
fig.data[50].visible = True

# Create and add fixed cost slider
fixed_cost_slider_steps = []
for idx in range(len(fig.data)):
    A = A_map[idx]
    visible_mask = [False] * len(fig.data)
    visible_mask[idx] = True
    step_config = dict(
        method="update",
        args=[{"visible": visible_mask},
              # {"title": f"Fixed cost = {A}"}
              ],
        label=f"{A}",
    )
    fixed_cost_slider_steps.append(step_config)

sliders = [dict(
    active=50,
    currentvalue={"prefix": "Fixed cost: "},
    pad={"t": 50},
    steps=fixed_cost_slider_steps
)]

fig.update_layout(
    sliders=sliders,
    xaxis_title="Days of year",
    yaxis_title="On-hand inventory",
    showlegend=True
)

fig.show()
fig.write_html("../data/eoq_plotly.html", full_html=False, include_plotlyjs=False)
"""
Diese Zeile vor dem exportierten code hinzufügen:
<script src="https://cdn.plot.ly/plotly-2.25.2.min.js" charset="utf-8"></script>
"""
