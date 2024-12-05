# Translated from breadth.m

import numpy as np

def breadth(CIJ, source):
    """
    Auxiliary function for breadthdist.

    Implementation of breadth-first search.

    Args:
        CIJ (numpy.ndarray): Binary (directed/undirected) connection matrix.
        source (int): Source vertex.

    Returns:
        tuple: A tuple containing:
            distance (numpy.ndarray): Distance between 'source' and i'th vertex (0 for source vertex).
            branch (numpy.ndarray): Vertex that precedes i in the breadth-first search tree (-1 for source vertex).

    Notes:
        Breadth-first search tree does not contain all paths (or all shortest paths), but allows the determination of at least one path with minimum distance. The entire graph is explored, starting from source vertex 'source'.
    """

    N = CIJ.shape[0]

    # Colors: white, gray, black
    white = 0
    gray = 1
    black = 2

    # Initialize colors
    color = np.zeros(N, dtype=int)
    # Initialize distances
    distance = np.full(N, np.inf)
    # Initialize branches
    branch = np.zeros(N, dtype=int)

    # Start on vertex 'source'
    color[source] = gray
    distance[source] = 0
    branch[source] = -1
    Q = [source]

    # Keep going until the entire graph is explored
    while Q:
        u = Q.pop(0)
        ns = np.where(CIJ[u, :] == 1)[0]  #Find neighbors

        for v in ns:
            # This allows the 'source' distance to itself to be recorded
            if distance[v] == 0:
                distance[v] = distance[u] + 1
            if color[v] == white:
                color[v] = gray
                distance[v] = distance[u] + 1
                branch[v] = u
                Q.append(v)
        color[u] = black

    return distance, branch

