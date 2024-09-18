import os
import sys

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt

from optimization.weber_problem import generate_data
from util.math_helper import eq


def objective_function(y: np.ndarray, weights, lons, lats, n_locs) -> float:
    """
    y contains first the longitudes, then the latitudes.
    """
    yx = y[:n_locs]
    yy = y[n_locs:]
    ret = 0.0
    for i, lon in enumerate(lons):
        lat = lats[i]
        mini = sys.maxsize
        for j in range(n_locs):
            dist = np.sqrt((lon - yx[j]) ** 2 + (lat - yy[j]) ** 2)
            if dist < mini:
                mini = dist
        ret += mini * weights[i]
    return ret


def optimize(weights, lons, lats, n_locs):
    start_lons = np.repeat(lons.mean(), n_locs)
    start_lats = np.repeat(lats.mean(), n_locs)
    x0 = np.concatenate((start_lons, start_lats))
    result = scipy.optimize.minimize(objective_function,
                                     x0=x0,
                                     args=(weights, lons, lats, n_locs),
                                     method='Powell')
    return result.x[: n_locs], result.x[n_locs:]


def get_assignments(lons, lats, x_j, y_j, n_locs):
    assert n_locs == len(x_j) == len(y_j)
    ret = np.repeat(-1, len(lons))
    for i, lon in enumerate(lons):
        lat = lats[i]
        mini = sys.maxsize
        argmin = -1
        for j in range(n_locs):
            dist = np.sqrt((lon - x_j[j]) ** 2 + (lat - y_j[j]) ** 2)
            if dist < mini:
                mini = dist
                argmin = j
        ret[i] = argmin
    return ret


def visualize_simple(weights, lons, lats, x_j, y_j, site_i, volume_j):
    for idx, lon in enumerate(lons):
        lat = lats[idx]
        plt.plot([x_j[site_i[idx]], lon], [y_j[site_i[idx]], lat], color='grey')
    plt.scatter(lons, lats, sizes=weights, zorder=3)
    plt.scatter(x_j, y_j, sizes=volume_j, zorder=2)
    plt.show()


def get_tables(weights, lons, lats, x_j, y_j, site_i, volume_j):
    columns = ['ID', 'lat', 'lon', 'type', 'color', 'volume']
    lst_site = [(f"S_{j}", y_j[j], x_j[j], 'site', '#ff3300', volume_j[j] * 100)
                for j in range(len(x_j))]
    df_site = pd.DataFrame(lst_site, columns=columns)
    df_site.set_index('ID', inplace=True)

    columns += ['site']
    lst_cust = [(f"C_{i}", lats[i], lons[i], 'customer', '#0044ff', weights[i] * 100, df_site.index[site_i[i]])
                for i in range(len(lons))]
    df_cust = pd.DataFrame(lst_cust, columns=columns)
    df_cust.set_index('ID', inplace=True)

    return df_site, df_cust


def visualize_streamlit(weights, lons, lats, x_j, y_j, site_i, volume_j):
    import streamlit as st
    df_site, df_cust = get_tables(weights, lons, lats, x_j, y_j, site_i, volume_j)
    df_cust.drop(columns=['site'], inplace=True)
    map_data = pd.concat((df_site, df_cust))
    st.map(map_data, latitude='lat', longitude='lon', color='color', size='volume')


def visualize_pydeck(weights, lons, lats, x_j, y_j, site_i, volume_j):
    import pydeck as pdk
    df_site, df_cust = get_tables(weights, lons, lats, x_j, y_j, site_i, volume_j)
    df_flows = pd.DataFrame(
        [(i, row['site'], row['lon'], row['lat'], row['volume'],
          df_site.loc[row['site'], 'lon'], df_site.loc[row['site'], 'lat'])
         for i, row in df_cust.iterrows()],
        columns=['customer', 'site', 'lon_customer', 'lat_customer', 'flow', 'lon_site', 'lat_site'])

    customer_layer = pdk.Layer(
        'ScatterplotLayer',
        df_cust,
        get_position=['lon', 'lat'],
        auto_highlight=True,
        get_radius=10000,  # meters
        get_fill_color=[180, 0, 200],  # RGB
        pickable=True)

    site_layer = pdk.Layer(
        'ScatterplotLayer',
        df_site,
        get_position=['lon', 'lat'],
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
        longitude=df_cust['lon'].mean(),
        latitude=df_cust['lat'].mean(),
        zoom=4,
        min_zoom=1,
        max_zoom=15)

    deck = pdk.Deck(layers=[flow_layer, customer_layer, site_layer], initial_view_state=view_state)
    deck.to_html('fig.html')


def check():
    n = 10
    ws, xs, ys = generate_data(n)
    assert eq(objective_function(y=np.concatenate([xs, ys]), weights=ws, lons=xs, lats=ys, n_locs=n), 0.0)


check()


def run():
    print(os.getcwd())
    np.random.seed(7)
    ws, xs, ys = generate_data(100)
    n_locs = 4
    x_j, y_j = optimize(ws, xs, ys, n_locs)
    obj_value = objective_function(np.concatenate((x_j, y_j)), ws, xs, ys, n_locs)
    print(f"Objective function value = {obj_value:.2f}")
    site_i = get_assignments(xs, ys, x_j, y_j, n_locs)
    volume_j = np.repeat(0.0, n_locs)
    for i, j in enumerate(site_i):
        volume_j[j] += ws[i]
    # visualize_simple(ws, xs, ys, x_j, y_j, site_i, volume_j)
    # visualize_streamlit(ws, xs, ys, x_j, y_j, site_i, volume_j)
    visualize_pydeck(ws, xs, ys, x_j, y_j, site_i, volume_j)


if __name__ == '__main__':
    run()
