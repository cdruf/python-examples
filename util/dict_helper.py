from collections import defaultdict
from functools import reduce
from typing import List, Dict, Tuple, TypeVar, Any, Set

import gurobipy as gp
import numpy as np

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


def group_tupledict_sum(dct: Dict[Tuple, Any], key_index: int):
    ret = defaultdict(float)
    for k, v in dct.items():
        ret[k[key_index]] += v
    return dict(ret)


def group_tupledict_mean(dct: Dict[Tuple, float | int], key_index: int):
    """Group the tuple-dictionary by one particular index of the key-tuples and calculate the group averages."""
    ret = defaultdict(float)
    counts = defaultdict(int)
    for k, v in dct.items():
        ret[k[key_index]] += v
        counts[k[key_index]] += 1
    return {k: v / counts[k] for k, v in ret.items()}


def group_tupledict_multi_sum(dct: Dict, key_indices: List[int]):
    ret = defaultdict(float)
    for k, v in dct.items():
        ret[tuple(k[idx] for idx in key_indices)] += v
    return dict(ret)


def group_dict_2d_by_1st_key_sum(dct: Dict[Any, Dict]):
    return {k: sum(d.values()) for k, d in dct}


def group_dict_2d_by_2nd_key_sum(dct: Dict[Any, Dict]):
    ret = defaultdict(int)
    for k1, d in dct.items():
        for k2, val in d.items():
            ret[k2] += val
    return dict(ret)


def merge_sum_dicts(d1, d2):
    """Alternative version with Counter"""
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


def to_np_1d_0(dct: Dict, n, mapping=None, dtype=np.int_) -> np.array:
    ret = np.full(n, 0, dtype=dtype)
    for k, val in dct.items():
        if mapping is not None:
            k = mapping[k]
        ret[k] = val
    return ret


def to_np_2d_0(dct: Dict[Tuple, Any], m, n, map1=None, map2=None) -> np.array:
    ret = np.full((m, n), 0, dtype=np.int_)
    for (k1, k2), val in dct.items():
        if map1 is not None:
            k1 = map1[k1]
        if map2 is not None:
            k2 = map2[k2]
        ret[k1, k2] = val
    return ret


def indicator_set_to_np_2d(s: Set[Tuple], m, n, map1=None, map2=None) -> np.array:
    ret = np.full((m, n), 0, dtype=np.int_)
    for (x, y) in s:
        if map1 is not None:
            x = map1[x]
        if map2 is not None:
            y = map2[y]
        ret[x, y] = 1
    return ret


def append_tuplelists(lists: List[list]):
    return gp.tuplelist(reduce(lambda x, y: x + y, lists, []))


def append_tupledicts(dicts: List[dict]):
    ret = {}
    for dct in dicts:
        ret.update(dct)
    return gp.tupledict(ret)


def map_keys(dct: Dict, mapping) -> Dict:
    if callable(mapping):
        return {mapping(k): v for k, v in dct.items()}
    return {mapping[k]: v for k, v in dct.items()}


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


def get_all_slices_2d(dtc_ij: Dict[Tuple, float]):
    ret_i: Dict[Tuple, float] = defaultdict(float)
    ret_j: Dict[Tuple, float] = defaultdict(float)
    ret = 0.0
    for (i, j), v in dtc_ij.items():
        ret_i[i] += v
        ret_j[j] += v
        ret += v
    return dict(ret_i), dict(ret_j), ret


def get_all_slices_3d(dtc_ijk: Dict[Tuple, float | int], dtype=float):
    ret_ij: Dict[Tuple, dtype] = defaultdict(dtype)
    ret_ik: Dict[Tuple, dtype] = defaultdict(dtype)
    ret_jk: Dict[Tuple, dtype] = defaultdict(dtype)
    ret_i: Dict[Tuple, dtype] = defaultdict(dtype)
    ret_j: Dict[Tuple, dtype] = defaultdict(dtype)
    ret_k: Dict[Tuple, dtype] = defaultdict(dtype)
    ret = 0.0
    for (i, j, k), v in dtc_ijk.items():
        ret_ij[(i, j)] += v
        ret_ik[(i, k)] += v
        ret_jk[(j, k)] += v
        ret_i[i] += v
        ret_j[j] += v
        ret_k[k] += v
        ret += v
    return (dict(ret_ij), dict(ret_ik), dict(ret_jk),
            dict(ret_i), dict(ret_j), dict(ret_k),
            ret)
