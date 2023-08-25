import numpy as np
import plotly.graph_objects as go

# Create figure
fig = go.Figure()

# Add traces, one for each slider step
for step in np.arange(0, 1, 0.1):
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color="#00CED1", width=6),
            name="𝜈 = " + str(step),
            x=np.arange(0, 1, 0.1),
            y=np.sin(step * np.arange(0, 10, 0.01))))

# Make 10th trace visible
fig.data[1].visible = True

# Create and add slider
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)

# fig.show()
import os

print(os.getcwd())
fig.write_html("../data/plotly_export.html", full_html=False, include_plotlyjs=False)
"""
Diese Zeile vor dem exportierten Stuff hinzufügen:
<script src="https://cdn.plot.ly/plotly-2.25.2.min.js" charset="utf-8"></script>

Funktion zum Exportieren:
fig.write_html("export.html", full_html=False, include_plotlyjs=False)"""
