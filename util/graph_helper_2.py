from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Any, Dict, Iterable, Set

import matplotlib.pyplot as plt
import numpy as np

from util.constants import WAIT, DEADHEAD
from util.priority_queue import MyMinPriorityQueue


@dataclass(init=False)
class MyDoublyWeightedDirectedMultigraph:
    """
    A directed multi-graph can have multiple arcs between the same source and destination vertices.
    Because of that property, we store the graph as an incidence list, instead of an adjacency list.
    We can identify each arc by its source, its destination, and an ID.

    """
    num_nodes: int  # Number of nodes. The nodes are identified by their index.
    num_arcs: int  # Number of arcs.
    a_i: Dict[int, Any]  # Can be used to add attributes to the nodes (index -> attribute)
    i_a: Dict[Any, int]  # If the attribute is hashable this can be used to obtain the node index by the attribute

    # Incidence list
    inc: List[Set[Tuple[int, int]]]  # node -> set of outgoing arcs identified by target node and an ID
    # ID: see constants

    cost_ijk: Dict[Tuple[int, int, int], float]  # from, to, ID -> cost
    energy_ijk: Dict[Tuple[int, int, int], float]  # from, to, ID -> energy

    @classmethod
    def start_empty(cls):
        ret = cls(num_nodes=0)
        return ret

    def __init__(self, num_nodes: int):
        """Create graph initializing the number of nodes and the adjacency list."""
        self.num_nodes = num_nodes
        self.num_arcs = 0
        self.a_i = {}
        self.i_a = {}
        self.inc = [set() for _ in range(self.num_nodes)]
        self.cost_ijk = {}
        self.energy_ijk = {}

    def add_node(self, attribute=None, hashable=True, fail_on_duplicate_attr=True) -> int:
        """
        Add a new node to the graph.

        :param attribute: An attribute that is stored for the node.
        :param hashable: Indicates if the attribute is a hashable value.
            If so the attribute and the node are added to the 'reverse' mapping of a_i, i_a.
        :param fail_on_duplicate_attr: If True, an error is raised in case an attribute value
            that already exists is supplied.

        :return: The index of the new node.
        """
        if attribute is not None:
            self.a_i[self.num_nodes] = attribute
            if hashable:
                if attribute in self.i_a:
                    if fail_on_duplicate_attr:
                        raise RuntimeError('Duplicate attribute!')
                    else:
                        print('WARNING: duplicate attribute')
                self.i_a[attribute] = self.num_nodes

        self.inc.append(set())
        self.num_nodes += 1
        return self.num_nodes - 1

    def add_arc(self, from_: int, to: int, id: int, cost: float, energy: float):
        """

        :param from_: Tail of arc.
        :param to: Head of arc.
        :param id: ID to make that arc identifiable (either requester ID, or the deadheading or waiting constants)
        :param cost: Cost for traversing arc.
        :param energy: Energy needed for traversing arc.
        """
        assert from_ < self.num_nodes
        assert to < self.num_nodes
        assert id in (DEADHEAD, WAIT) or id >= 0  # see constants
        assert (to, id) not in self.inc[from_]

        self.inc[from_].add((to, id))
        self.cost_ijk[from_, to, id] = cost
        self.energy_ijk[from_, to, id] = energy
        self.num_arcs += 1


def get_adjacency_lists_from_adjacency_matrix(matrix, fill=True, eps=0.001):
    """
    Calculates the adjacency lists.

    :param matrix: Square indicator matrix.
    :param fill: If True, the indices without any neighbor are included with an empty list.
    :param eps: Numerical value for zero for floating point comparisons.
    return
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


def plot_graph(nodes: Dict[Any, Tuple[float, float]], edges: Iterable[Tuple[int, int]], node_annotations=True):
    """
    Paint a graph.

    Args:
        nodes: Node names (keys) and their positions (values).
        edges: List of edges.
        node_annotations: Indicates if the node numbers are displayed.
    """
    xs, ys = zip(*nodes.values())
    plt.scatter(xs, ys, s=50, alpha=0.5)
    if node_annotations:
        for idx, node in enumerate(nodes.keys()):
            plt.text(xs[idx], ys[idx], node)
    for i, j in edges:
        plt.plot([nodes[i][0], nodes[j][0]], [nodes[i][1], nodes[j][1]], color='grey')
    plt.xlim((min(xs), max(xs)))
    plt.ylim((min(ys), max(ys)))
    plt.tight_layout()
    plt.xticks(np.arange(min(xs), max(xs), step=1))
    plt.yticks(np.arange(min(ys), max(ys), step=1))
    plt.show()


def plot_my_graph(g: MyDoublyWeightedDirectedMultigraph):
    nodes = {i: g.a_i[i] for i in range(g.num_nodes)}  # get time-space pairs as coordinates of nodes
    edges = [(i, j) for i in range(g.num_nodes) for (j, _) in g.inc[i]]
    plot_graph(nodes=nodes, edges=edges)


def dijkstra(num_nodes: int,
             adj: List[Iterable[int]] | Tuple[Iterable[int]],
             source: int,
             distances_ij: dict) -> Tuple[np.array, np.array]:
    """
    Get the shortest paths from a source node to all nodes.
    :param num_nodes: The number of nodes of the graph.
    :param adj: Adjacency list representation of the graph.
    :param source: The source node.
    :param distances_ij: The distance function to be used.
    :return:
    """
    assert 0 <= source < num_nodes
    dists = np.repeat(np.inf, num_nodes)
    dists[source] = 0.0
    preds = np.repeat(-1, num_nodes)  # -1 representing none
    finished = set()  # finished nodes
    queue = MyMinPriorityQueue()  # candidate queue - pairs of distance and node
    queue.add(item=source, prio=0.0)

    while not queue.empty():
        u, _ = queue.get()
        finished.add(u)
        for v in adj[u]:
            if dists[v] > dists[u] + distances_ij[u, v]:
                dists[v] = dists[u] + distances_ij[u, v]
                preds[v] = u
                queue.add_or_update(item=v, prio=dists[v])

    return dists, preds


def dijkstra_labels_to_route(source, destination, predecessor_i):
    """
    Transform the labels to a route.

    :param source: The source node, which must the one Dijkstra has been run with.
    :param destination: The destination node.
    :param predecessor_i: The predecessors determined by Dijkstra.
    :return: A route represented by a sequence of nodes.
    """
    # build route from source to dest
    node = destination
    lst = []
    while node != source:
        lst.append(node)
        node = predecessor_i[node]
    lst.append(source)
    lst.reverse()
    return tuple(lst)


def dijkstra_origin_destination(num_nodes: int,
                                adj: List[Iterable[int]] | Tuple[Iterable[int]],
                                source: int,
                                dest: int,
                                distances_ij: dict) -> Tuple[Tuple, float]:
    """
    :param num_nodes: The nodes of the graph are assumed to be numbered from 0 to num_nodes - 1.
    :param adj: The adjacency list.
    :param distances_ij: distances between nodes.
    :param source: Source node.
    :param dest: Destination node.


    :return:
        Sequence of nodes.
        Total distance.
    """
    assert 0 <= source < num_nodes
    assert 0 <= dest < num_nodes
    if source == dest:
        return (), 0.0

    dists, preds = dijkstra(num_nodes, adj, source, distances_ij)
    route = dijkstra_labels_to_route(source=source, destination=dest, predecessor_i=preds)
    return route, dists[dest]


def get_all_pair_shortest_paths(num_nodes: int,
                                adj: List[Iterable[int]] | Tuple[Iterable[int]],
                                distances_ij: dict) -> Tuple[Dict, Dict]:
    dists_s = {}
    preds_s = {}
    for src in range(num_nodes):
        dists, preds = dijkstra(num_nodes, adj, src, distances_ij)
        dists_s[src] = dists
        preds_s[src] = preds

    return dists_s, preds_s


if __name__ == "__main__":
    g_test = MyDoublyWeightedDirectedMultigraph.start_empty()
    for ii in range(20):
        g_test.add_node(attribute=str(ii))

    print("End")
