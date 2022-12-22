#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a gurobi implementation of the Giapetto problem from
Winston - Operations Research: Applications and Algorithms
including linear programming sensitivity analysis.
"""
import gurobipy as grb
import numpy as np

from util import gurobi_helper as helper

# %%
# Problem parameter
n = 2  # index j
b = [100, 80, 40]
m = len(b)
A = np.array([[2, 1],
              [1, 1],
              [1, 0]])
c = [3, 2]

# %%
# Model in normal form

model = grb.Model("giapetto_model")
x = model.addVars(range(n), name='x')
s = model.addVars(range(m), name='s')
model.setObjective(x.prod(c), grb.GRB.MAXIMIZE)
model.addConstrs((grb.quicksum(A[i][j] * x[j] for j in range(n)) + s[i] == b[i] for i in range(m)), "c")
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

# print('\nRHS Ranges')
# df_rhs_ranges = helper.get_rhs_ranges(model)
# print(df_rhs_ranges)
