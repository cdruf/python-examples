from typing import Mapping, List, Any, Tuple, Dict
from collections import defaultdict
import pandas as pd

EPS = 1e-6  # Numerical value for zero


def eq(x, y, tolerance=EPS):
    return abs(x - y) <= tolerance


def is_binary(xs: pd.Series):
    return xs.apply(lambda x: eq(x, 0) or eq(x, 1)).all()


def get_adjacency_lists_from_adjacency_matrix(matrix):
    """
    Calculates the adjacency lists
    :param matrix: Square indicator matrix.
    :return
        ret1: key is the 1st dimension of the matrix.
        ret2: key is the 2nd dimension of the matrix.
    """
    ret1 = defaultdict(list)
    ret2 = defaultdict(list)
    for i, x in enumerate(matrix):
        for j, y in enumerate(x):
            if eq(y, 1):
                ret1[i].append(j)
                ret2[j].append(i)
    return dict(ret1), dict(ret2)


def get_adjacency_lists_from_arcs(arcs: List[Tuple[Any, Any]]) -> Tuple[Dict[Any, List[Any]], Dict[Any, List[Any]]]:
    """[(1, 2), (1, 3), (4, 3)] maps to
    ({1: [2, 3], 4:[3]},
    {2: [1], 3: [1, 4]})
    """
    ret_forwards = defaultdict(list)
    ret_backwards = defaultdict(list)
    for x, y in arcs:
        ret_forwards[x].append(y)
        ret_backwards[y].append(x)
    return dict(ret_forwards), dict(ret_backwards)


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
