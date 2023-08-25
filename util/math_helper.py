from collections import defaultdict
from typing import Mapping, List, Any, TypeVar, Set

import pandas as pd
from scipy.stats import norm

# Declare type variable
T = TypeVar('T')

EPS = 1e-6  # Numerical value for zero


def eq(x, y, tolerance=EPS) -> bool:
    return abs(x - y) <= tolerance


def is_binary(xs: pd.Series) -> bool:
    return xs.apply(lambda x: eq(x, 0) or eq(x, 1)).all()


def get_image(x_maps_to_y: Mapping[Any, T]) -> List[T]:
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


def loss_function_standard_normal(x: float) -> float:
    return norm.pdf(x) - x * (1 - norm.cdf(x))


def loss_function_normal(x: float, mu: float, sigma: float) -> float:
    return sigma * loss_function_standard_normal((x - mu) / sigma)


if __name__ == "__main__":
    result = loss_function_normal(120, 100, 50)
    print(result)  # 11.52 expected
