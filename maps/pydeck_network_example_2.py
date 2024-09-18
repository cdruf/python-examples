import sys

import pandas as pd
import pydeck as pdk

from maps.pydeck_scatter_example_2 import generate_data
from util.distance_helper import distance_haversine

weights, lons, lats = generate_data(300)
idx = [f"C_{i}" for i in range(len(weights))]
df_customers = pd.DataFrame({'ID': idx, 'longitude': lons, 'latitude': lats, 'weight': weights})
df_customers.set_index('ID', inplace=True)

_, lons, lats = generate_data(6)
idx = [f"S_{j}" for j in range(len(lons))]
df_sites = pd.DataFrame({'ID': idx, 'longitude': lons, 'latitude': lats})
df_sites.set_index('ID', inplace=True)


def get_closest(lon, lat, df_sites):
    argmin = -1
    mini = sys.maxsize
    for idx, row in df_sites.iterrows():
        dist = distance_haversine(lat, lon, row['latitude'], row['longitude'])
        if dist < mini:
            mini = dist
            argmin = idx
    return argmin


df_customers['site'] = df_customers[['longitude', 'latitude']].apply(
    lambda x: get_closest(x['longitude'], x['latitude'], df_sites), axis=1)

df_sites['volume'] = df_customers.groupby('site')['weight'].sum()
df_sites['volume'] = df_sites['volume'].fillna(0.0)

df_flows = pd.DataFrame(
    [(i, row['site'], row['longitude'], row['latitude'], row['weight'],
      df_sites.loc[row['site'], 'longitude'], df_sites.loc[row['site'], 'latitude'])
     for i, row in df_customers.iterrows()],
    columns=['customer', 'site', 'lon_customer', 'lat_customer', 'flow', 'lon_site', 'lat_site'])

customer_layer = pdk.Layer(
    'ScatterplotLayer',
    df_customers,
    get_position=['longitude', 'latitude'],
    auto_highlight=True,
    get_radius=10000,  # meters
    get_fill_color=[180, 0, 200],  # RGB
    pickable=True)

site_layer = pdk.Layer(
    'ScatterplotLayer',
    df_sites,
    get_position=['longitude', 'latitude'],
    auto_highlight=True,
    get_radius=100000,  # meters
    get_fill_color=[0, 255, 200],  # RGB
    pickable=True)

# Flow layer
flow_layer = pdk.Layer(
    'LineLayer',
    df_flows,
    opacity=0.5,
    getWidth='weight',
    widthScale=1,
    getSourcePosition=['lon_site', 'lat_site'],
    getTargetPosition=['lon_customer', 'lat_customer'],
    getColor=[0, 150, 100],
)

# Set the viewport location
view_state = pdk.ViewState(
    longitude=df_customers['longitude'].mean(),
    latitude=df_customers['latitude'].mean(),
    zoom=6,
    min_zoom=1,
    max_zoom=15)

deck = pdk.Deck(layers=[flow_layer, customer_layer, site_layer], initial_view_state=view_state)
deck.to_html('fig.html')
