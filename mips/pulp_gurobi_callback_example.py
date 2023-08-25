"""
Created on 2023-08-24
@author: Christian Ruf
"""

import pulp as pl
from gurobipy import GRB


# Define callback
def mycallback(model, where):
    if where == GRB.Callback.MIP:
        # General MIP callback
        nodecnt = model.cbGet(GRB.Callback.MIP_NODCNT)
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        solcnt = model.cbGet(GRB.Callback.MIP_SOLCNT)
        if nodecnt - model._lastnode >= 100:
            model._lastnode = nodecnt
            actnodes = model.cbGet(GRB.Callback.MIP_NODLFT)
            itcnt = model.cbGet(GRB.Callback.MIP_ITRCNT)
            cutcnt = model.cbGet(GRB.Callback.MIP_CUTCNT)
            print(f'\nMIP-Callback: {nodecnt}, {actnodes}, {itcnt}, {objbst}, {objbnd}, {solcnt}, {cutcnt}')

    elif where == GRB.Callback.MIPSOL:
        # MIP solution callback
        nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        solcnt = model.cbGet(GRB.Callback.MIPSOL_SOLCNT)
        # x = model.cbGetSolution(model._vars)  # <<< Does not work with PuLP
        print(f'\n\n\nMIPSOL-Callback **** New solution at node {nodecnt}, obj {obj}, sol {solcnt}\n\n')
        if solcnt == 1:
            print("\t2 solutions really are enough => terminate!")
            model.terminate()


# Build model withb PuLP

mdl = pl.LpProblem("MyLP", pl.LpMaximize)
xs = pl.LpVariable.dicts('x', indices=range(2), cat=pl.LpInteger)
mdl += pl.lpSum((i + 1) * x for i, x in xs.items())
mdl += pl.lpSum(x for x in xs.values()) <= 5, "constraint_1"
mdl += xs[1] <= 4, "constraint_2"
print(mdl)

# Solve with callback
solver = pl.GUROBI(mipgap=0.001, timeLimit=600)
# mdl.solve(solver)
# solver.buildSolverModel(mdl)
# mdl.solverModel.optimize(mycallback)
mdl.solve(solver, callback=mycallback)

# Get solution
print("Status =", pl.LpStatus[mdl.status])
print("Objective value =", pl.value(mdl.objective))
for variable in mdl.variables():
    print(f"{variable.name} = {variable.varValue}")
