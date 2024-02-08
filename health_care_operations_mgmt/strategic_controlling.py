#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HCOM: exercise 1.3, strategic controlling
"""
import gurobipy as grb



try:
    
    # Parameter
    n = 3 # number of DRGs (Index j)
    m = 3 # number of resources (Index i)
    K_i = [1000, 2000, 2000]
    c_ij = [[0.4, 0.6, 1.5], [1.0, 1.3, 2.0], [6.0, 5.0, 10.0]]
    d_j = [1700, 1000, 4000]
    a_j = [1300, 1100, 3000]
    
    # Create a new model
    model = grb.Model("model")
    
    # Create variables
    x_j = []
    for j in range(n):
        x_j.append(model.addVar(vtype = grb.GRB.INTEGER, name = "x_" + str(j)))

    # Set objective
    expr = 0
    for j in range(n):
        expr += (d_j[j] - a_j[j]) * x_j[j] 
    model.setObjective(expr, grb.GRB.MAXIMIZE)

    # Add constraint
    for i in range(m):
        expr = 0
        for j in range(n):
            expr += c_ij[i][j] * x_j[j]
        model.addConstr(expr <= K_i[i], "c_" + str(i))
    
    x_j[1].lb = 100

    model.optimize()

    # Print results
    print('\nSolution')
    for v in model.getVars():
        print('%s %g' % (v.varName, v.s))
    print('Obj: %g' % model.objVal)
    
    # LP-Relaxation
    print('\nLP-Relaxation')
    model.Params.outputFlag = 0
    for v in model.getVars():
        v.vtype = grb.GRB.CONTINUOUS
    model.optimize()
    
    print('\nvar \t val \t RC')
    for v in model.getVars():
        print('%s \t %g \t %g' % (v.varName, v.s, v.rc))
    
    print('\nconstr \t slack or surplus \t dual prices')
    for c in model.getConstrs():
        print('%s \t %g \t %g' % (c.constrName, c.slack, c.pi))
    

except grb.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')


