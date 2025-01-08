import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def create_figure():
    """Create an exemplary interactive chart. """
    ret = go.Figure()

    # Add one trace for each slider step
    steps = list(range(1, 101))
    for step in steps:
        xs = np.arange(0, 2 * math.pi, 0.01)
        ys = np.sin(xs - step / 100)
        ret.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                visible=False,
                line=dict(color="#00CED1", width=6),
                name=f"shift = {step / 10:.2f}"
            ))

    # Make 1st trace visible
    ret.data[0].visible = True

    # Create and add slider
    slider_steps = []
    for idx, data in enumerate(ret.data):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(ret.data)},
                  {"title": f"Slider switched to step {idx}"}],
        )
        step["args"][0]["visible"][idx] = True  # Toggle i'th trace to "visible"
        slider_steps.append(step)

    ret.update_layout(
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Current value: "},
            pad={"t": 50},
            steps=slider_steps)],
        yaxis_range=[-1.1, 1.1])

    return ret


if __name__ == '__main__':
    fig = create_figure()
    choice = input("Press 1 to showing the plot and 2 for storing the plot.")
    if choice == "1":
        fig.show()
    else:
        data_folder = Path(__file__).parent.parent / "data"
        fig.write_html(data_folder / "plotly_export.html", full_html=False, include_plotlyjs=False)
        """
        Either export including the plotly JavaScript or without. 
        If without, add this line to the website:
        <script src="https://cdn.plot.ly/plotly-2.25.2.min.js" charset="utf-8"></script>
        """
