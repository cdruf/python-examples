"""
Created on 2023-02-27
@author: Christian Ruf
"""

import pulp as pl

mdl = pl.LpProblem("MyLP", pl.LpMaximize)
xs = pl.LpVariable.dicts('x', indices=range(2), cat=pl.LpBinary)
mdl += pl.lpSum(x for x in xs.values())
mdl += pl.lpSum(x for x in xs.values()) <= 1, "constraint_1"
mdl += xs[1] >= 1, "constraint_2"

print(mdl)

solver = pl.GUROBI()
solver.buildSolverModel(mdl)

x0 = mdl.solverModel.getVarByName('x_0')
x1 = mdl.solverModel.getVarByName('x_1')
print(x0.Start)
print(x1.Start)

x0.Start = 1
x1.setAttr('Start', 0)
mdl.solverModel.update()
print(x0.Start)
print(x1.Start)

mdl.solve(solver)

print(pl.LpStatus[mdl.status])
print(pl.value(mdl.objective))

for variable in mdl.variables():
    print(f"{variable.name} = {variable.varValue}")
