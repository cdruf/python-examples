#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import gurobipy as grb



try:
    
    # Parameter
    n = 3 # j
    m = 4 # i
    b = [48, 20, 8, 5]
    A = [[8, 6, 1], \
            [4, 2, 1.5], \
            [2, 1.5, 0.5], \
            [0, 1, 0]]
    c = [60, 30, 20]
    
    # Create a new model
    model = grb.Model("model")
    
    # Create variables
    x = []
    for j in range(n):
        x.append(model.addVar(vtype = grb.GRB.CONTINUOUS, name = "x_" + str(j)))

    # Set objective
    expr = 0
    for j in range(n):
        expr += c[j] * x[j] 
    model.setObjective(expr, grb.GRB.MAXIMIZE)

    # Add constraints
    for i in range(m):
        expr = 0
        for j in range(n):
            expr += A[i][j] * x[j]
        model.addConstr(expr <= b[i], "c_" + str(i))
    
    model.optimize()

    # Print results
    print('\nSolution')
    
    print('\nObj: %g' % model.objVal)
    
    print('\nvar \t val \t RC')
    for v in model.getVars():
        print('%s \t %g \t %g' % (v.varName, v.x, v.rc))

    print('\nconstr \t slack or surplus \t dual prices')
    for c in model.getConstrs():
        print('%s \t %g \t %g' % (c.constrName, c.slack, c.pi))
    
    
    print('\nRanges in which the basis is unchanged')
        
    print('\nObjective coefficient Ranges')
    print('var \t coeff \t allowable increase \t allowable decrease')
    for v in model.getVars():
        if -0.001 <= v.x <= 0.001: # nicht in der Basis
            allowableIncrease = -v.rc
            allowableDecrease = grb.GRB.INFINITY
        else:
            allowableIncrease = 0
            allowableDecrease = 0
        print('%s \t %g \t %g \t %g' % (v.varName, v.obj, \
                                        allowableIncrease, allowableDecrease))
    
    print('\nRHS Ranges')
    print('constr \t RHS \t allowable increase \t allowable decrease')
    for c in model.getConstrs():
        if -0.001 <= c.slack <= 0.001: # kein Slack => bindend
            if c.sense == '<':
                allowableIncrease = 0
                allowableDecrease = 0
            elif c.sense == '>':
                allowableIncrease = 0
                allowableDecrease = 0 
            else:
                allowableIncrease = 0
                allowableDecrease = 0
        else:
            if c.sense == '<':
                allowableIncrease = grb.GRB.INFINITY 
                allowableDecrease = c.slack
            elif c.sense == '>':
                allowableIncrease = c.slack
                allowableDecrease == -grb.GRB.INFINITY 
            else:
                allowableIncrease = 0
                allowableDecrease = 0
            print("TODO")
        print('%s \t %g \t %g \t %g' % (c.constrName, c.rhs, \
                                        allowableIncrease, allowableDecrease))

except grb.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')


