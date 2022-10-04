import json

import pydeck

f = open("./vancouver-blocks.json")
data_geojson = json.load(f)

INITIAL_VIEW_STATE = pydeck.ViewState(
    latitude=49.254,
    longitude=-123.13,
    zoom=11,
    max_zoom=16,
)

gg = pydeck.Layer(
    'GeoJsonLayer',
    data_geojson,
    stroked=False,
    filled=True,
    extruded=True,
    wireframe=True,
    pickable=True,
    get_line_color=[255, 255, 255]

)

fig = pydeck.Deck(
    layers=[gg],
    initial_view_state=INITIAL_VIEW_STATE,
)

fig.to_html("fig.html")
fig.show()
