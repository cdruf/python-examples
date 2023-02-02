from collections import defaultdict
from functools import reduce
from typing import List, Dict, Tuple, TypeVar, Any

import gurobipy as gp

# Declare type variable
T = TypeVar('T')
T1 = TypeVar('T1')
T2 = TypeVar('T2')
T3 = TypeVar('T3')


def print_first_entries(dct, n=3, sep=',', sort_keys=False):
    if sort_keys:
        tmp = sorted(list(dct.items()), key=lambda x: x[0])
    else:
        tmp = list(dct.items())
    print(sep.join([f"{k} -> {v}" for k, v in tmp[:min(n, len(tmp))]]))


def sum_2d_dict_values(dct: Dict[Any, Dict[Any, T]]) -> T:
    return sum(sum(d.values()) for d in dct.values())


def sum_2d_tupledict_by_dimension(dct: Dict[Tuple, T], dim):
    if dim == 1:
        keys = set(k for k, _ in dct)
        return {k: sum([val for (k1, _), val in dct.items() if k1 == k]) for k in keys}
    elif dim == 2:
        keys = set(k for _, k in dct)
        return {k: sum([val for (_, k2), val in dct.items() if k2 == k]) for k in keys}
    else:
        raise RuntimeError("dim must be 1 or 2")


def merge_sum_dicts(d1, d2):
    return {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1) | set(d2)}


def dict_2d_to_tupledict(dct: dict):
    return gp.tupledict({(k1, k2): val for k1, inner in dct.items()
                         for k2, val in inner.items()})


def tupledict_to_2d_dict(dct: Dict[Tuple[T1, T2], T]) -> Dict[T1, Dict[T2, T]]:
    ret = defaultdict(dict)
    for (k1, k2), val in dct.items():
        ret[k1][k2] = val
    return dict(ret)


def tupledict_to_2d_dict(dct: Dict[Tuple[T1, T2], T]) -> Dict[T1, Dict[T2, T]]:
    ret = defaultdict(dict)
    for (k1, k2), val in dct.items():
        ret[k1][k2] = val
    return dict(ret)


def tupledict_to_3d_dict(dct: Dict[Tuple[T1, T2, T3], T]) -> Dict[T1, Dict[T2, Dict[T3, T]]]:
    ret = {}
    for (k1, k2, k3), val in dct.items():
        if k1 not in ret:
            ret[k1] = {}
        if k2 not in ret[k1]:
            ret[k1][k2] = {}
        ret[k1][k2][k3] = val
    return ret


def append_tuplelists(lists: List[list]):
    return gp.tuplelist(reduce(lambda x, y: x + y, lists, []))


def append_tupledicts(dicts: List[dict]):
    ret = {}
    for dct in dicts:
        ret.update(dct)
    return gp.tupledict(ret)


def group_dict_sum(dct: Dict, key_index: int):
    ret = defaultdict(float)
    for k, v in dct.items():
        ret[k[key_index]] += v
    return dict(ret)


def group_dict_sum_multi(dct: Dict, key_indices: List[int]):
    ret = defaultdict(float)
    for k, v in dct.items():
        ret[tuple(k[idx] for idx in key_indices)] += v
    return dict(ret)


class Tupledict:
    def __init__(self, dct):
        self.dct = dct
        self.dim = len(list(dct.keys())[0])

    def sum(self, *args):
        if len(args) == 0:
            return sum(self.dct.values())

        if len(args) != self.dim:
            raise ValueError(f"Number of arguments must either be zero or equal to the number of dimensions {self.dim}")

        return sum(v for k, v in self.dct.items() if True)  # TODO: idea: overwrite comparison
