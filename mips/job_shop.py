from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import pulp
from pulp import lpSum

from pathlib import Path
from util.data_helper import load_data_and_clean
from util.math_helper import get_pre_images, eps

DATA_FOLDER = Path(__file__).parent.parent / "data"

# %%

machines_df = load_data_and_clean((DATA_FOLDER / "job_shop_example_data.xlsx"),
                                  "Machines", "ID", skiprows=0)

M = machines_df.index.to_list()

jobs_df = load_data_and_clean((DATA_FOLDER / "job_shop_example_data.xlsx"),
                              "Jobs", "ID", skiprows=0)
J = jobs_df.index.to_list()
T_j = jobs_df['due date'].to_dict()

operations_df = load_data_and_clean((DATA_FOLDER / "job_shop_example_data.xlsx"),
                                    "Operations", "ID", skiprows=0)
O = operations_df.index.to_list()
j_o = operations_df['job'].to_dict()
m_o = operations_df['machine'].to_dict()
D_o = operations_df['duration'].to_dict()

precendences_df = load_data_and_clean((DATA_FOLDER / "job_shop_example_data.xlsx"),
                                      "Precedences", ["Operation 1", "Operation 2"], skiprows=0)
P = precendences_df.index.to_list()
Delta_o1o2 = precendences_df['time lag'].to_dict()
j_p = precendences_df['job'].to_dict()

# %%

O_j = get_pre_images(j_o, J)
M_j = {j: set([m_o[o] for o in O_j[j]]) for j in J}
O_m = get_pre_images(m_o, M)
P_j = get_pre_images(j_p, J)

# %%

model = pulp.LpProblem('ProductionPlanningModel', pulp.LpMinimize)
print(datetime.now())

# Variables
t_o = pulp.LpVariable.dicts(
    name='t', indexs=O, cat=pulp.LpContinuous, lowBound=0)
y_idxs = [(o1, o2) for m in M for o1 in O_m[m] for o2 in O_m[m] if o1 != o2]
y_o1o2 = pulp.LpVariable.dicts(
    name='y', indexs=y_idxs, cat=pulp.LpBinary)
s_j = pulp.LpVariable.dicts(
    name='s', indexs=J, cat=pulp.LpContinuous, lowBound=0)

print(f"Number of t-variables = {len(t_o)}")
print(f"Number of y-variables = {len(y_o1o2)}")
print(f"Number of s-variables = {len(s_j)}")

# %%
# Objective function
model += lpSum([s_j[j] for j in J])

# Constraints
# Precedence constraints
for j in J:
    for o1, o2 in P_j[j]:
        model += t_o[o1] + Delta_o1o2[o1, o2] <= t_o[o2], f"c_precedence_{j},{o1},{o2}"

# Order constraints
for o1, o2 in y_o1o2:
    if o1 < o2:
        model += y_o1o2[o1, o2] + y_o1o2[o2, o1] == 1, f"c_order_{o1},{o2}"

# Machine occupation constraints
total_duration = sum([D_o[o] for o in O])
for o1, o2 in y_o1o2:
    if o1 != o2:
        model += t_o[o1] + D_o[o1] <= t_o[o2] + total_duration * (1 - y_o1o2[o1, o2]), \
                 f"c_machine_occupation_{o1},{o2}"

# Tardiness constraints
for j in J:
    for o in O_j[j]:
        t_o[o] + D_o[o] - T_j[j] <= s_j[j], f"c_tardiness_{j},{o}"

# %%

# Optimize
print('Solve model')
# model.writeLP(DATA_FOLDER / "production_planning.lp")
model.solve()
status = model.status
if status == pulp.LpStatusInfeasible:
    print("Model is infeasible")
elif status == pulp.LpStatusNotSolved:
    print("Model not optimal")
elif status == pulp.LpStatusOptimal:
    print("Model optimal")

# %%
# Get variable values & create dictionaries
tval = {o: t_o[o].varValue for o in O}
sval = {j: s_j[j].varValue for j in J}

# %%

# Create table for job overview
for j in J:
    sorted_ops = sorted(O_j[m], key=lambda x: tval[x])
    out = ' --> '.join([f"{o} [{tval[o]:.0f} - {tval[o] + D_o[o]:.0f}]" for o in sorted_ops])
    print(f"Job {j} tardiness = {tval[j]:.0f}, schedule: {out}")

# Create table for machine overview
for m in M:
    sorted_ops = sorted(O_m[m], key=lambda x: tval[x])
    out = ' --> '.join([f"{o} [{tval[o]:.0f} - {tval[o] + D_o[o]:.0f}]" for o in sorted_ops])
    print(f"Machine {m} schedule: {out}")
