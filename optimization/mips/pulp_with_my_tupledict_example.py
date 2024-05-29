"""
Created on 2023-01-31
@author: Christian Ruf

"""
from itertools import product

import pulp as pl

from util.dict_helper import Tupledict

mdl = pl.LpProblem("My_LP_Problem", pl.LpMaximize)
xs = Tupledict(pl.LpVariable.dicts('x', indices=product(range(3), range(2)), lowBound=0, cat='Continuous'))
print(type(xs))
print(xs)

# Objective function
mdl += xs.sum(), "Z"

###

# Constraints
mdl += xs.sum() <= 5, "constraint_1"
mdl += xs.sum(0, '*') <= 2, "constraint_2"
mdl += xs.sum('*', 1) <= 1, "constraint_3"

# print(mdl)
mdl.solve()

print(pl.LpStatus[mdl.status])
print(pl.value(mdl.objective))

for variable in mdl.variables():
    print(f"{variable.name} = {variable.varValue}")
