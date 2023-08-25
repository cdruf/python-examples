import numpy as np
import plotly.graph_objects as go

from inventory_management.economic_order_quantity import EOQ

A_min = 20
A_max = 100
v_min = 1
v_max = 10

eoqs = {(a, v): EOQ(D=12000, A=a, v=v, r=0.1) for a in range(A_min, A_max + 1) for v in range(v_min, v_max + 1)}
Q_max = 0
for eoq in eoqs.values():
    eoq.compute()
    if eoq.Q > Q_max:
        Q_max = eoq.Q

# Create figure
fig = go.Figure(layout_yaxis_range=[0, Q_max])

idx_map = {}
idx = 0
A_v_map = {}

# Add traces, one for each slider step
for fixed_cost in np.arange(A_min, A_max + 1, 1):
    for unit_variable_cost in np.arange(v_min, v_max + 1, 1):
        idx_map[fixed_cost, unit_variable_cost] = idx
        A_v_map[idx] = (fixed_cost, unit_variable_cost)
        idx += 1
        xs, ys = eoqs[fixed_cost, v_min].get_xs_and_ys()
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#00CED1", width=6),
                name=f"A = {fixed_cost}, v={unit_variable_cost}",
                x=xs,
                y=ys))

# Make 1st trace visible
fig.data[0].visible = True

# Create and add fixed cost slider
fixed_cost_slider_steps = []
for idx in range(len(fig.data)):
    A, v = A_v_map[idx]
    visible_mask = [False] * len(fig.data)
    visible_mask[idx] = True
    step_config = dict(
        method="update",
        args=[{"visible": visible_mask},
              {"title": f"Fixed cost = {A}, unit variable cost = {v}"}],
        #label=f"{eoqs[(idx + A_min, v_min)].A}",
    )
    fixed_cost_slider_steps.append(step_config)

sliders = [
    dict(
        active=0,
        currentvalue={"prefix": "Fixed cost: "},
        pad={"t": 50},
        steps=fixed_cost_slider_steps
    ),
    dict(
        active=0,
        currentvalue={"prefix": "Unit variable cost: "},
        pad={"t": 150},
        steps=fixed_cost_slider_steps
    )]

fig.update_layout(
    sliders=sliders
)

fig.show()
# fig.write_html("..data/eoq_plotly.html", full_html=False, include_plotlyjs=False)
"""
Diese Zeile vor dem exportierten code hinzufügen:
<script src="https://cdn.plot.ly/plotly-2.25.2.min.js" charset="utf-8"></script>
"""
