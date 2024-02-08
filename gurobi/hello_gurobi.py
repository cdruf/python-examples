#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import gurobipy as gb
from gurobipy import GRB


def mycallback(model, where):
    if where == GRB.Callback.MIPSOL:
        print("\n\n*** New solution ***\n")


# Create a new model
m = gb.Model("mini_mip")

# Create variables
x = m.addVar(vtype=GRB.BINARY, name="x")

# Set objective
m.setObjective(x, GRB.MAXIMIZE)

# Add constraint: x <= 4
m.addConstr(x <= 4, "c0")

m.optimize(callback=mycallback)

for v in m.getVars():
    print('%s %g' % (v.varName, v.s))

print('Obj: %g' % m.objVal)
