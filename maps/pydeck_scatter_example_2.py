import numpy as np
import pandas as pd
import pydeck


def generate_data(n, min_weight=1, max_weight=3, min_lon=-117, max_lon=-77, min_lat=32, max_lat=41):
    alpha = np.random.rand(n)
    weights = min_weight + alpha * (max_weight - min_weight)
    alpha = np.random.rand(n)
    lons = min_lon + alpha * (max_lon - min_lon)
    alpha = np.random.rand(n)
    lats = min_lat + alpha * (max_lat - min_lat)
    return weights, lons, lats


weights, lons, lats = generate_data(100)
df = pd.DataFrame({'longitude': lons, 'latitude': lats, 'weights': weights})
print(df.head())

scatter_layer = pydeck.Layer('ScatterplotLayer', df,
                             get_position=['longitude', 'latitude'],
                             get_radius=10000,  # meters
                             auto_highlight=True,
                             get_fill_color=[180, 0, 200, 140],  # Set an RGBA value for fill
                             pickable=True)
initial_view_state = pydeck.ViewState(
    latitude=lats.mean(),
    longitude=lons.mean(),
    zoom=6,
    min_zoom=4,
    max_zoom=7,
)
fig = pydeck.Deck(
    layers=[scatter_layer],
    initial_view_state=initial_view_state,
)
fig.to_html("fig.html")
