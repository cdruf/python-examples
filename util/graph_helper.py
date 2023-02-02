from collections import defaultdict


def get_adjacency_lists_from_adjacency_matrix(matrix, fill=True, eps=0.001):
    """
    Calculates the adjacency lists
    :param matrix: Square indicator matrix.
    :param fill: If True, the indices without any neighbor are included with an empty list.
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
