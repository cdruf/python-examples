# -*- coding: utf-8 -*-
"""
Facility location problem with PuLP.
@author: Christian Ruf
"""
from itertools import product
from pathlib import Path

import numpy as np
import pulp as pl
from matplotlib import pyplot as plt

from util.data_helper import load_data_and_clean

EPS = 0.0001  # floating point precision


def distance_approx(lon_1, lat_1, lon_2, lat_2):
    return 111 * np.sqrt((lon_1 - lon_2) ** 2 + (lat_1 - lat_2) ** 2)


def _sum(coefficients, variables):
    return pl.lpSum(coefficients[idx] * var for idx, var in variables.items())


def get_variable_values(variables):
    return {idx: pl.value(var) for idx, var in variables.items()}


# Load data
path = Path('../../data/Facility_location_data.xlsx')
customers_df = load_data_and_clean(path, "Customers", "Customer", to_lower_case=False)
sites_df = load_data_and_clean(path, "Sites", "Site", to_lower_case=False)

print("Customer table: ")
print(customers_df.head())

print("\nSite table: ")
print(sites_df.head())

# Extract customer data
customers = customers_df.index.tolist()
lat_c = customers_df["Latitude"].to_dict()
lon_c = customers_df["Longitude"].to_dict()
demand_c = customers_df["Demand"].to_dict()

# Extract site data
sites = sites_df.index.tolist()
status_s = sites_df["Status"].to_dict()
lat_s = sites_df["Latitude"].to_dict()
lon_s = sites_df["Longitude"].to_dict()
fixed_cost_s = sites_df["Fixed cost"].to_dict()
capacity_s = sites_df["Capacity"].to_dict()
unit_production_cost_s = sites_df["Unit production cost"].to_dict()
unit_shipping_cost_per_km_s = sites_df["Unit shipping cost per mile"].to_dict()

# Pre-compute
flow_cost_per_unit_sc = {
    (s, c): distance_approx(lon_s[s], lat_s[s], lon_c[c], lat_c[c]) * unit_shipping_cost_per_km_s[s]
    for s, c in product(sites, customers)}

# Build the model
m = pl.LpProblem("FLP", pl.LpMinimize)

# Variables
y_s = pl.LpVariable.dicts(name="y", indices=sites, cat=pl.LpBinary)
x_sc = pl.LpVariable.dicts(name="x", indices=product(sites, customers), cat=pl.LpContinuous, lowBound=0.0)

# Objective
m += _sum(fixed_cost_s, y_s) + _sum(flow_cost_per_unit_sc, x_sc)

# Constraints
for c in customers:
    m += pl.lpSum(x_sc[s, c] for s in sites) >= demand_c[c], f"c_demand_{c}"
for s in sites:
    m += pl.lpSum(x_sc[s, c] for c in customers) <= capacity_s[s], f"c_capacity_{s}"

m.writeLP("./FLP.lp")

# Solve
m.solve()
print(m.status)
print(pl.value(m.objective))

# Get variable values
y_s_vals = get_variable_values(y_s)
x_sc_vals = get_variable_values(x_sc)

for idx, val in y_s_vals.items():
    if val > 0.0:
        print(f"{idx}: {val}")
for idx, val in x_sc_vals.items():
    if val > 0.0:
        print(f"{idx}: {val}")

# Visualize
production_quantity_s = {s: sum(x_sc_vals[s, c] for c in customers) for s in sites}

site_volumes = [production_quantity_s[s] / 10 for s in sites]
colors = [idx for idx, s in enumerate(sites)]
fig, ax = plt.subplots()
ax.scatter(lon_c.values(), lat_c.values(), c=['black'] * len(customers), s=[5] * len(customers))
ax.scatter(lon_s.values(), lat_s.values(), c=colors, s=site_volumes)
for s, c in product(sites, customers):
    if x_sc_vals[s, c] > 0:
        plt.plot([lon_c[c], lon_s[s]], [lat_c[c], lat_s[s]])
ax.grid(True)
plt.show()
