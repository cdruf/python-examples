#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing pydeck map plotting.

@author: Christian
"""
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

# USA mainland approximate boundaries
north_tip_lat = 45
south_tip_lat = 33
east_tip_lon = -80
west_tip_lon = -116

# Generate locations with demands
df = pd.DataFrame(
    np.random.rand(100, 2)
    * [north_tip_lat - south_tip_lat, east_tip_lon - west_tip_lon]
    + [south_tip_lat, west_tip_lon],
    columns=['lat', 'lon'])
df['demand'] = np.random.randint(0, 100, size=len(df))

# Generate some flows between locations
df_from = df.loc[np.random.rand(len(df)) < 0.1, ['lat', 'lon']]
df_to = df.loc[np.random.rand(len(df)) < 0.1, ['lat', 'lon']]
df_from['dummy'] = 1
df_to['dummy'] = 1
flows = df_from.merge(df_to, on='dummy')
flows.drop(columns=['dummy'], inplace=True)
flows['quantity'] = np.random.randint(10, high=20, size=len(flows))

# Location layer
location_layer = pdk.Layer(
    'ScatterplotLayer',
    df,
    auto_highlight=True,
    opacity=0.9,
    stroked=True,
    filled=True,
    get_position=['lon', 'lat'],  # 1st value = longitude column header
    get_radius='demand',  # radius is given in meters
    radius_scale=1000,
    get_fill_color=[180, 0, 200, 140],  # set an RGBA value for fill
)

# Flow layer
flow_layer = pdk.Layer(
    'LineLayer',
    flows,
    opacity=0.9,
    getWidth='quantity',
    widthScale=0.1,
    getSourcePosition=['lon_x', 'lat_x'],
    getTargetPosition=['lon_y', 'lat_y'],
    getColor=[180, 0, 200, 140],
)

# Set the viewport location
view_state = pdk.ViewState(
    longitude=df['lon'].mean(),
    latitude=df['lat'].mean(),
    zoom=2,
    min_zoom=2,
    max_zoom=20,
    pitch=4.5,
    bearing=0)

# Combined all of it and render a viewport
deck = pdk.Deck(layers=[flow_layer, location_layer],
                initial_view_state=view_state,
                map_style='mapbox://styles/mapbox/light-v9')

deck.to_html('deck-example.html')  # does not show map only markers

st.pydeck_chart(deck)
