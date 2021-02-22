# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:59:14 2021

@author: 49856
"""

import pulp

my_lp_problem = pulp.LpProblem("My_LP_Problem", pulp.LpMaximize)

x = pulp.LpVariable('x', lowBound=0, cat='Continuous')
y = pulp.LpVariable('y', lowBound=2, cat='Continuous')

# Objective function
my_lp_problem += 4 * x + 3 * y, "Z"

# Constraints
my_lp_problem += 2 * y <= 25 - x
my_lp_problem += 4 * y >= 2 * x - 8
my_lp_problem += y <= 2 * x - 5

print(my_lp_problem)


my_lp_problem.solve()

print(pulp.LpStatus[my_lp_problem.status])
print(pulp.value(my_lp_problem.objective))

for variable in my_lp_problem.variables():
    print("{} = {}".format(variable.name, variable.varValue))