import gurobipy as gp
from gurobipy import GRB

mdl = gp.Model()

# Variables
x = mdl.addVar(lb=-1, ub=4, name="x")
twox = mdl.addVar(lb=-2, ub=8, name="2x")
sinx = mdl.addVar(lb=-1, ub=1, name="sinx")
cos2x = mdl.addVar(lb=-1, ub=1, name="cos2x")
expx = mdl.addVar(name="expx")

# Objective
mdl.setObjective(sinx + cos2x + 1, GRB.MINIMIZE)

# Linear constraints
lc1 = mdl.addConstr(0.25 * expx - x <= 0)
lc2 = mdl.addConstr(2.0 * x - twox == 0)

# sinx = sin(x)
gc1 = mdl.addGenConstrSin(x, sinx, "gc1")
# cos2x = cos(twox)
gc2 = mdl.addGenConstrCos(twox, cos2x, "gc2")
# expx = exp(x)
gc3 = mdl.addGenConstrExp(x, expx, "gc3")

# Optimize
# mdl.params.FuncNonlinear = 1
# mdl.optimize()

# Optimize
mdl.update()
gc1.funcnonlinear = 1
gc2.funcnonlinear = 1
gc3.funcnonlinear = 1
mdl.optimize()
