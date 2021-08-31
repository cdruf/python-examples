#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import sys
from gurobipy import *

try:

    # Create a new model
    m = Model("mini_mip")

    # Create variables
    x = m.addVar(vtype=GRB.BINARY, name="x")

    # Set objective
    m.setObjective(x, GRB.MAXIMIZE)

    # Add constraint: x <= 4
    m.addConstr(x <= 4, "c0")

    m.optimize()

    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % m.objVal)

except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')
