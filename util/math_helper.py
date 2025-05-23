import math
from collections import defaultdict
from itertools import chain, combinations
from typing import Mapping, List, Any, TypeVar, Set

import numpy as np
import pandas as pd
from scipy.linalg import circulant
from scipy.stats import norm

# Declare type variable
T = TypeVar('T')

EPS = 1e-9  # Numerical value for zero


def eq(x, y, tolerance=EPS) -> bool:
    return abs(x - y) <= tolerance


def is_binary(xs: pd.Series) -> bool:
    return xs.apply(lambda x: eq(x, 0) or eq(x, 1)).all()


def get_image(x_maps_to_y: Mapping[Any, T]) -> List[T]:
    return list(set([y for x, y in x_maps_to_y.items()]))


def get_pre_images_depr(x_maps_to_y: Mapping, Y: list = None):
    X_y = {y: [] for y in set(x_maps_to_y.values())}
    if Y is not None:
        X_y.update({y: [] for y in Y})
    for x, y in x_maps_to_y.items():
        X_y[y].append(x)
    return X_y


def get_pre_images(x_maps_to_y: Mapping, Y: list | set = None):
    """
    Given  mapping f : X --> Y,
    returns a mapping g : Y --> Sets of X, g(y) = X_y
    where X_y = {x | f(x) = y}.

    It possible to supply Y, which is then used to include y that are not in the image of f.

    Args:
        x_maps_to_y (Map): A mapping X --> Y
        Y (List): Set of elements without an x that should be included anyways.

    Returns:
        "Reverse mapping"

    """
    X_y = {y: set() for y in set(x_maps_to_y.values())}
    if Y is not None:
        X_y.update({y: set() for y in Y})
    for x, y in x_maps_to_y.items():
        assert x not in X_y[y]
        X_y[y].add(x)
    return X_y


def reverse_n_to_m(m: Mapping[Any, Set]):
    """
    Given is mapping m: X --> Y where Y is a set of sets.
    Let Z = union of all Y.
    Create the mapping ret: Z --> W where W is a set of sets, defined by
    ret(z) = w = {x | z in m(x)}.
    """
    expanded = [(x, y) for x, ys in m.items() for y in ys]
    ret = defaultdict(set)
    for x, y in expanded:
        ret[y].add(x)
    return dict(ret)


def powerset(iterable, non_empty=True):
    """
    Return the powerset.
    """
    start = 1 if non_empty else 0
    return chain.from_iterable(combinations(iterable, r) for r in range(start, len(iterable) + 1))


def loss_function_standard_normal(x: float) -> float:
    return norm.pdf(x) - x * (1 - norm.cdf(x))


def loss_function_normal(x: float, mu: float, sigma: float) -> float:
    return sigma * loss_function_standard_normal((x - mu) / sigma)


def ceil(x: float, digits=0) -> int | float:
    if digits == 0:
        return math.ceil(x)
    return math.ceil(x * 10 ** digits) / 10 ** digits


def floor(x: float, digits=0) -> int | float:
    if digits == 0:
        return math.floor(x)
    return math.floor(x * 10 ** digits) / 10 ** digits


def get_circular_smoothing_matrix(n: int, m: int):
    """
    :param n: Length of the vector to be smoothed.
    :param m: Number of periods to be convoluted.
    """
    column = np.array([1 / m] * math.ceil(m / 2) + [0] * (n - m) + [1 / m] * math.floor(m / 2))
    ret = circulant(column)
    assert all(abs(ret.sum(axis=0) - 1.0) <= 1e-6)
    return ret


def get_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return np.mean(np.abs((actual - predicted) / actual))


if __name__ == "__main__":
    result = loss_function_normal(120, 100, 50)
    print(result)  # 11.52 expected
