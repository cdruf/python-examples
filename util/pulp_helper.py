import pulp as pl


def get_positive_var_values(var_dict, EPS=0.001):
    return {k: var.varValue for k, var in var_dict.items() if var.varValue > EPS}


def get_positive_expr_values(dct, EPS=0.001):
    return {k: pl.value(expr) for k, expr in dct.items() if pl.value(expr) > EPS}


def get_positive_expr_values_int(dct, tolerance=0.001):
    ret = {}
    for k, expr in dct.items():
        val = pl.value(expr)
        assert abs(val - round(val)) <= tolerance
        if val > tolerance:
            ret[k] = round(val)
    return ret


def get_binary_variable_values(var_dict, tolerance=0.001, only_ones=False):
    ret = {}
    for k, var in var_dict.items():
        val = pl.value(var)
        if abs(val - round(val)) > tolerance:
            raise RuntimeError('Value not binary')
        if only_ones:
            if round(val) == 1:
                ret[k] = round(val)
        else:
            ret[k] = round(val)
    return ret
