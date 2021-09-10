# -*- coding: utf-8 -*-
"""


"""
import math

import gurobipy as gp

inf_pos = float('inf')
inf_neg = float('-inf')


def both_inf_and_same_sign(value1, value2):
    if not math.isinf(value1):  # tests for positive or negative infinity
        return False
    if not math.isinf(value2):
        return False
    # Both values are either positive or negative infinity
    if not value1 * value2 > 0:
        return False
    return True


def eq(f1, f2):
    return abs(f1 - f2) < 0.00001


def compare_2_constraints(constraint1, constraint2):
    """Not possible to compare coefficients"""
    if not both_inf_and_same_sign(constraint1.rhs, constraint2.rhs) and not eq(constraint1.rhs, constraint2.rhs):
        raise RuntimeError(f"Constraints' {constraint1.ConstrName} RHS differ")
    if not constraint1.Sense == constraint2.Sense:
        raise RuntimeError(f"Constraints' {constraint1.ConstrName} Sense differ")
    # print(constraint1)
    # print(constraint2)


def compare_all_constraints(constraints1, model2):
    n_comparisons = 0
    for constraint1 in constraints1:
        name1 = constraint1.ConstrName
        constraint2 = model2.getConstrByName(name1)
        if constraint2 is None:
            raise RuntimeError(f"Constraint {name1} is missing in model 2")
        compare_2_constraints(constraint1, constraint2)
        n_comparisons += 1
    print(f"{n_comparisons} constraints compared")


def compare_2_variables(variable1, variable2):
    if not both_inf_and_same_sign(variable1.LB, variable2.LB) and not eq(variable1.LB, variable2.LB):
        raise RuntimeError(f"Variables' {variable1.VarName} LBs differ")
    if not both_inf_and_same_sign(variable1.UB, variable2.UB) and not eq(variable1.UB, variable2.UB):
        raise RuntimeError(f"Variables' {variable1.VarName} UBs differ")
    if not eq(variable1.Obj, variable2.Obj):
        raise RuntimeError(f"Variables' {variable1.VarName} Objs differ")
    if variable1.VType != variable2.VType:
        raise RuntimeError(f"Variables' {variable1.VarName} VTypes differ")


def compare_all_variables(variables1, model2):
    n_comparisons = 0
    for variable1 in variables1:
        name1 = variable1.VarName
        variable2 = model2.getVarByName(name1)
        if variable2 is None:
            raise RuntimeError(f"Variable {name1} is missing in model 2")
        compare_2_variables(variable1, variable2)
        n_comparisons += 1
    print(f"{n_comparisons} variables compared")


def compare_matrix(model1, variables1, constraints1, model2):
    n_comparisons = 0
    for constraint1 in constraints1:
        constraint_name1 = constraint1.ConstrName
        constraint2 = model2.getConstrByName(constraint_name1)
        if constraint2 is None:
            raise RuntimeError(f"Constraint {constraint_name1} is missing in model 2")
        for variable1 in variables1:
            variable_name1 = variable1.VarName
            variable2 = model2.getVarByName(variable_name1)
            if variable2 is None:
                raise RuntimeError(f"Variable {variable2} is missing in model 2")
            coefficient1 = model1.getCoeff(constraint1, variable1)
            coefficient2 = model2.getCoeff(constraint2, variable2)
            if not eq(coefficient1, coefficient2):
                raise RuntimeError(f"The coefficients of constraint {constraint_name1} "
                                   f"and variable {variable_name1} differ - "
                                   f"{coefficient1} <> {coefficient2}")
            n_comparisons += 1
    print(f"{n_comparisons} coefficients compared")
    print(f"N constraints x N varialbes = {len(constraints1) * len(variables1)}")


def compare_objectives(objective1, objective2):
    """Compare the linear expression of two objectives.
    Variable coefficients are already covered via variables"""
    if objective1.size() != objective2.size():
        raise RuntimeError("Objectives have different many terms")
    if objective1.getConstant() != objective2.getConstant():
        raise RuntimeError("Objectives have different constants")


# Main

file1 = "/tmp/m1.lp"  # Hauptmodel
file2 = "/tmp/m2.lp"

# Read models
m1 = gp.read(file1)
m2 = gp.read(file2)
m1.update()
m2.update()

# Get variables
vars1 = m1.getVars()
vars2 = m2.getVars()
print(f"N variables model 1 = {len(vars1)}")
print(f"N variables model 2 = {len(vars2)}")

# Get constraints
constrs1 = m1.getConstrs()
constrs2 = m2.getConstrs()
print(f"N constraints model 1 = {len(constrs1)}")
print(f"N constraints model 2 = {len(constrs2)}")

# Get objectives
obj1 = m1.getObjective()
obj2 = m2.getObjective()

# Compare
compare_all_variables(vars1, m2)
compare_all_constraints(constrs1, m2)
compare_matrix(m1, vars1, constrs1, m2)
compare_objectives(m1.getObjective(), m2.getObjective())

print("\n\n=== Models are equal ===")
