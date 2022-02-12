import gurobipy as gp


def print_first_entries(dct, n=3, sep=',', sort_keys=False):
    if sort_keys:
        tmp = sorted(list(dct.items()), key=lambda x: x[0])
    else:
        tmp = list(dct.items())
    print(sep.join([f"{k} -> {v}" for k, v in tmp[:min(n, len(tmp))]]))


def sum_2d_dict_by_dimension(dct, dim):
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


def append_tuplelists(lists: List[list]):
    return gp.tuplelist(reduce(lambda x, y: x + y, lists, []))


def append_tupledicts(dicts: List[dict]):
    ret = {}
    for dct in dicts:
        ret.update(dct)
    return gp.tupledict(ret)
