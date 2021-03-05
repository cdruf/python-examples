#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple assignment model.

Tuples for multi-dimensional subscripts
Dictionaries for indexed dat
Gurobi tuplelist: for storing a list of tuples
"""

import gurobipy as grb


def solve(categories, minNutrition, maxNutrition, foods,
          cost, nutritionValues):
    m = grb.Model("diet")

    # Create variables
    nutrition = {}
    for c in categories:
        nutrition[c] = m.addVar(lb=minNutrition[c], ub=maxNutrition[c], name=c)
        
    buy = {}
    for f in foods:
        buy[f] = m.addVar(obj=cost[f], name=f)
    
    # Set objective
    m.modelSense = grb.GRB.MINIMIZE
    
    # set nutrition for all categories
    for c in categories:
        m.addConstr(nutrition[c] == 
                    grb.quicksum(nutritionValues[f,c] * buy[f] for f in foods),
                    c)
    
    m.optimize()
    
    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))
    
    print('Obj: %g' % m.objVal)