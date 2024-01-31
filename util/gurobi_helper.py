import gurobipy as grb
import numpy as np
import pandas as pd

eps = 0.001  # floating point precision


def get_basic_variables(model):
    return filter(lambda x: x.vBasis == 0, model.getVars())


def get_non_basic_variables(model):
    return filter(lambda x: x.vBasis != 0, model.getVars())


def get_vbasis_name(variable):
    d = {0: 'basic', -1: 'non-basic at lower bound', -2: 'non-basic at upper bound', -3: 'super-basic'}
    return d[variable.vBasis]


def get_allowable_increase_of_objective_coefficient(model, variable):
    """
    Get the allowable increase of the objective coefficient of a variable such that the basis remains the same.
    If an objective function coefficient is changed the current solution remains feasible.
    The variable values and the objective function value may change though.

    For a non-basic variable, the reduced cost is the amount by which the coefficient must be improved
    before the LP has an optimal solution with the variable in the basis.
    For a basic variable, the allowable increase is ... .
    """
    if model.modelSense == 1:  # minimization
        if variable.vBasis != 0:  # non-basic
            return np.inf  # worsening the coefficient of a non-basic variable will not make it basic
        else:
            return variable.SAObjUp - variable.obj
    else:  # maximization
        if variable.vBasis != 0:  # non-basic
            return -variable.rc
        else:
            return variable.SAObjUp - variable.obj


def get_allowable_decrease_of_objective_coefficient(model, variable):
    if model.modelSense == 1:  # minimization
        if variable.vBasis != 0:  # non-basic
            return -variable.rc
        else:
            return variable.obj - variable.SAObjLow
    else:  # maximization
        if variable.vBasis != 0:  # non-basic
            return np.inf  # worsening the coefficient of a non-basic variable will not make it basic
        else:
            return variable.obj - variable.SAObjLow


def get_variables_df(model):
    """
    Get the variables with values and reduced costs in a DataFrame.
    """
    return pd.DataFrame(data=[(v.varName, v.x, (v.x * v.obj),
                               get_vbasis_name(v), v.rc,
                               v.SAObjLow, v.SAObjUp,
                               get_allowable_decrease_of_objective_coefficient(model, v),
                               get_allowable_increase_of_objective_coefficient(model, v))
                              for v in model.getVars()],
                        columns=['Variable', 'Value', 'Objective contribution',
                                 'Basis status', 'Reduced cost',
                                 'smallest obj. same basis', 'largest obj. same basis',
                                 'allowable decrease obj.', 'allowable increase obj.'])


def get_constraints_df(model):
    """
    Get the constraints with slacks and shadow prices in a DataFrame.
    """
    return pd.DataFrame(data=[(c.constrName, c.slack, c.pi) for c in model.getConstrs()],
                        columns=['Constraint', 'Slack or surplus', 'Dual price'])


# %%


def get_rhs_allowable_increase(constraint):
    """
    TODO: add it to constraint df
    """
    if -0.001 <= constraint.slack <= 0.001:  # no slack => binding
        return 0
    else:
        return (grb.GRB.INFINITY if constraint.sense == '<' else
                (constraint.slack if constraint.sense == '>' else 0))


def get_rhs_allowable_decrease(constraint):
    if -0.001 <= constraint.slack <= 0.001:  # no slack => binding
        return 0
    else:
        if constraint.sense == '<':
            return constraint.slack
        elif constraint.sense == '>':
            return -grb.GRB.INFINITY
        else:
            return 0


def get_rhs_ranges(model):
    return pd.DataFrame(
        data=[(c.constrName, c.rhs, get_rhs_allowable_increase(c), get_rhs_allowable_decrease(c))
              for c in model.getConstrs()],
        columns=['Constraint', 'RHS', 'Allowable increase', 'Allowable decrease'])


def identify_constraints_with_extreme_coefficients(mdl: grb.Model, lb=10e-4, ub=10e9):
    """Helper function to improve numeric properties of the model.

    Function is slow and is therefore only useful for debugging."""
    print(f"Detect extreme coefficients with lb = {lb} and ub = {ub}")
    mdl.update()
    for c in mdl.getConstrs():
        c_name = c.ConstrName
        for v in mdl.getVars():
            try:
                v_name = v.VarName
            except UnicodeDecodeError:
                v_name = "?"
            coefficient = mdl.getCoeff(c, v)
            if (coefficient != 0.0 and abs(coefficient) < lb) or abs(coefficient) > ub:
                print(f"WARNING: extreme coefficient for {c_name}, {v_name}, value = {coefficient}")


def identify_rhs_with_extreme_value(mdl: grb.Model, lb=1e-4, ub=1e4):
    for c in mdl.getConstrs():
        val = c.rhs
        if (val != 0.0 and abs(val) < lb) or abs(val) > ub:
            print(f"WARNING: extreme coefficient for {c.ConstrName}, value = {val}")
