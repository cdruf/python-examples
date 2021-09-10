# -*- coding: utf-8 -*-
"""
Script reads gurobi model from file in order to calculate the irreducible inconsistent subsystem of constraints.

@author: Christian Ruf
"""

import sys
import os
import gurobipy as gp
from gurobipy import GRB


print(os.getcwd())
file = "/home/christian/Dropbox/OR/Facility location - blog post 1/flp_infeasible.lp"

# Read and solve model

model = gp.read(file)
model.optimize()

if model.status == GRB.INF_OR_UNBD:
    # Turn presolve off to determine whether model is infeasible
    # or unbounded
    model.setParam(GRB.Param.Presolve, 0)
    model.optimize()

if model.status == GRB.OPTIMAL:
    print('Optimal objective: %g' % model.objVal)
    model.write('model.sol')
    sys.exit(0)
elif model.status != GRB.INFEASIBLE:
    print('Optimization was stopped with status %d' % model.status)
    sys.exit(0)

# Model is infeasible - compute an Irreducible Inconsistent Subsystem (IIS)

print('')
print('Model is infeasible')
model.computeIIS()
model.write("model.ilp")
print("IIS written to file 'model.ilp'")
