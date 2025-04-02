# -*- coding: utf-8 -*-
"""
Created in 2021

@author: Christian Ruf
"""

import pulp

mdl = pulp.LpProblem("My_LP_Problem", pulp.LpMaximize)

x = pulp.LpVariable('x', lowBound=0, cat='Continuous')
y = pulp.LpVariable('y', lowBound=2, cat='Continuous')

# Objective function
mdl += 4 * x + 3 * y, "Z"

# Constraints
mdl += 2 * y + x <= 25, "c_1"
mdl += x <= 5, "c_2"
mdl += y <= 4, "c_3"

print(mdl)


def solve():
    mdl.solve()
    print(pulp.LpStatus[mdl.status])
    print(pulp.value(mdl.objective))
    for variable in mdl.variables():
        print("{} = {}".format(variable.name, variable.varValue))


solve()
print("\n")

# Drop constraint & re-solve
del mdl.constraints['c_3']
solve()
print("\n")

# Replace objective & re-solve
mdl.objective = 2 * x + 2 * y
solve()
