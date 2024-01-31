#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an implementation of the Farmer Jones problem from
Winston - Operations Research: Applications and Algorithms.
"""

import pulp

mdl = pulp.LpProblem("farmer_jones_model", pulp.LpMaximize)

x1 = pulp.LpVariable('x1', lowBound=0, cat='Continuous')
x2 = pulp.LpVariable('x2', lowBound=0, cat='Continuous')

# Objective function
mdl += 3 * x1 + 4 * x2, "z"

# Constraints
mdl += x1 >= 30, "min_qty"
mdl += 1 / 10 * x1 + 1 / 25 * x2 <= 7, "surface"
mdl += 4 / 10 * x1 + 10 / 25 * x2 <= 40, "work"

# mdl += x1 == 50, "hack_1"
# mdl += x1 == 50, "hack_2"

print(mdl)

mdl.solve()

print('\n=== Solution ===\n')

print(pulp.LpStatus[mdl.status])
print(pulp.value(mdl.objective))
for variable in mdl.variables():
    print("{} = {}".format(variable.name, variable.varValue))
