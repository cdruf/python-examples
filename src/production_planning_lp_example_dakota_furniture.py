#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a gurobi implementation of the Dakota furniture model from
Winston - Operations Research: Applications and Algorithms
including linear programming sensitivity analysis.
"""
import gurobipy as grb
import numpy as np
import pandas as pd

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

print('\n=== Solution ===\n')
print('Objective value = %g' % model.objVal)

df_variables = pd.DataFrame(data=[(v.varName, v.x, v.rc) for v in model.getVars()],
                            columns=['Variable', 'Value', 'Reduced cost'])
print(df_variables)

df_constraints = pd.DataFrame(data=[(c.constrName, c.slack, c.pi) for c in model.getConstrs()],
                              columns=['Constraint', 'Slack or surplus', 'Dual price'])
print(df_constraints)

# %%

print('\n=== Ranges in which the basis is unchanged ===\n')

print('Objective coefficient ranges')
df_obj_coeff_ranges = pd.DataFrame(
    data=[(v.varName, v.obj,
           (-v.rc if abs(v.x) < 0.001 else 0),  # condition <==> not in basis
           (grb.GRB.INFINITY if abs(v.x) < 0.001 else 0))
          for v in model.getVars()],
    columns=['Variable', 'Objective coefficient', 'allowable increase', 'allowable decrease'])
print(df_obj_coeff_ranges)

print('\nRHS Ranges')


def get_rhs_allowable_increase(constraint):
    if -0.001 <= constraint.slack <= 0.001:  # no slack => binding
        return 0
    else:
        return (grb.GRB.INFINITY if constraint.sense == '<' else
                (constraint.slack if constraint.sense == '>' else 0))


def get_rhs_allowable_decrease(constraint):
    if -0.001 <= constraint.slack <= 0.001:  # no slack => binding
        return 0
    else:
        if constraint.sense == '<':
            return constraint.slack
        elif constraint.sense == '>':
            return -grb.GRB.INFINITY
        else:
            return 0


df_rhs_ranges = pd.DataFrame(
    data=[(c.constrName, c.rhs, get_rhs_allowable_increase(c), get_rhs_allowable_decrease(c))
          for c in model.getConstrs()],
    columns=['Constraint', 'RHS', 'Allowable increase', 'Allowable decrease'])
print(df_rhs_ranges)
