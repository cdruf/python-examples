from typing import Mapping

import pandas as pd

eps = 1e-6  # Numerical value for zero


def eq(x, y, tolerance=eps):
    return abs(x - y) <= tolerance


def is_binary(xs: pd.Series):
    return xs.apply(lambda x: eq(x, 0) or eq(x, 1)).all()


def get_image(x_maps_to_y: Mapping):
    return list(set([y for x, y in x_maps_to_y.items()]))


def get_pre_images(x_maps_to_y: Mapping, Y: list = None):
    """
    Given  mapping f : X --> Y,
    returns a mapping g : Y --> Sets of X, g(y) = X_y
    where X_y = {x | f(x) = y}.

    It possible to supply Y, which is then used to include y that are not in the image of f.

    Args:
        x_maps_to_y (Map): A mapping of the form X --> Y
        Y (List): Set of elements without an x that should be included anyways.

    Returns:
        "Reverse mapping"

    """
    X_y = {y: [] for y in set(x_maps_to_y.values())}
    if Y is not None:
        X_y.update({y: [] for y in Y})
    for x, y in x_maps_to_y.items():
        X_y[y].append(x)
    return X_y
