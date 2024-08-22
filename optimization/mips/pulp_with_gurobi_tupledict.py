"""
Created on 2023-01-31
@author: Christian Ruf

It's a bit of an ugly hack to avoid the constant of the expression to be the wrong type.

"""
from itertools import product

import gurobipy as gb
import pulp as pl


def dsum(dct, *args):
    ret = dct._sum(*args)
    ret.constant = 0.0
    return ret


mdl = pl.LpProblem("My_LP_Problem", pl.LpMaximize)

xs = gb.tupledict(pl.LpVariable.dicts('x', indices=product(range(3), range(2)), lowBound=0, cat='Continuous'))

# Objective function
s = xs.sum()
s.constant = 0.0
mdl += s, "Z"  # using gurobipy's sum function

# Constraints
mdl += s <= 5, "constraint_1"
mdl += dsum(xs, 0, '*') <= 2, "constraint_2"

print(mdl)
mdl.solve()

print(pl.LpStatus[mdl.status])
print(pl.value(mdl.objective))

for variable in mdl.variables():
    print(f"{variable.name} = {variable.varValue}")
