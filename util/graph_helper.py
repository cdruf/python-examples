from collections import defaultdict
from typing import List, Tuple, Any, Dict, Iterable

import matplotlib.pyplot as plt


def get_adjacency_lists_from_adjacency_matrix(matrix, fill=True, eps=0.001):
    """
    Calculates the adjacency lists
    :param matrix: Square indicator matrix.
    :param fill: If True, the indices without any neighbor are included with an empty list.
    :param eps: Numerical value for zero for floating point comparisons.
    :return
        ret1: key is the 1st dimension of the matrix.
        ret2: key is the 2nd dimension of the matrix.
    """
    if fill:
        ret1 = {i: [] for i in range(len(matrix))}
        ret2 = {j: [] for j in range(len(matrix[0]))}
        for i, row in enumerate(matrix):
            for j, y in enumerate(row):
                if 1 - eps <= y <= 1 + eps:
                    ret1[i].append(j)
                    ret2[j].append(i)
        return ret1, ret2

    # Do not fill
    ret1 = defaultdict(list)
    ret2 = defaultdict(list)
    for i, row in enumerate(matrix):
        for j, y in enumerate(row):
            if 1 - eps <= y <= 1 + eps:
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


def plot_graph(nodes: Dict[Any, Tuple[float, float]], edges: Iterable[Tuple[float, float]]):
    """
    Paint a graph.

    Args:
        nodes: Node names (keys) and their positions (values).
        edges: List of edges.
    """
    xs, ys = zip(*nodes.values())
    plt.scatter(xs, ys, s=50, alpha=0.5)
    for idx, node in enumerate(nodes.keys()):
        plt.text(xs[idx], ys[idx], node)
    for i, j in edges:
        plt.plot([nodes[i][0], nodes[j][0]], [nodes[i][1], nodes[j][1]], color='grey')
    plt.show()


if __name__ == '__main__':
    nodes = {'a': (1, 2), 'b': (1, 7), 'c': (5, 3)}
    edges = [('a', 'b'), ('b', 'c'), ('c', 'a')]
    plot_graph(nodes, edges)
