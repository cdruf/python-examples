import pulp as pl


def get_positive_var_values(var_dict, EPS=0.001):
    return {k: var.varValue for k, var in var_dict.items() if var.varValue > EPS}


def get_positive_expr_values(dct, EPS=0.001):
    return {k: pl.value(expr) for k, expr in dct.items() if pl.value(expr) > EPS}
