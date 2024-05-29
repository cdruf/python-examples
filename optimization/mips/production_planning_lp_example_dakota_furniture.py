#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a gurobi implementation of the Dakota furniture model from
Winston - Operations Research: Applications and Algorithms
including linear programming sensitivity analysis.
"""
import gurobipy as grb
import numpy as np

from util import gurobi_helper as helper

# %%


# Parameter
n = 3  # index j
m = 4  # index i
b = [48, 20, 8, 5]
A = np.array([[8, 6, 1],
              [4, 2, 1.5],
              [2, 1.5, 0.5],
              [0, 1, 0]])
c = [60, 30, 20]

# %%

model = grb.Model("model")
x = model.addVars(range(n), name="x")
model.setObjective(grb.quicksum(c[j] * x[j] for j in range(n)), grb.GRB.MAXIMIZE)
model.addConstrs((grb.quicksum([A[i][j] * x[j] for j in range(n)]) <= b[i] for i in range(m)), "c")
model.update()

# %%

model.optimize()

# %%

print('\n=== Solution ===\n')
print('Objective value = %g' % model.objVal)

df_variables = helper.get_variables_df(model)
print(df_variables, '\n')

df_constraints = helper.get_constraints_df(model)
print(df_constraints)

# %%


print('\nRHS Ranges')
df_rhs_ranges = helper.get_rhs_ranges(model)
print(df_rhs_ranges)
