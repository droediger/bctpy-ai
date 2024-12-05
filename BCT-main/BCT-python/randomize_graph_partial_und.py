# Translated from randomize_graph_partial_und.m

import numpy as np

def randomize_graph_partial_und(A, B, maxswap):
    """Swap edges with preserved degree sequence.

    Args:
        A: Undirected adjacency matrix.
        B: Edges to avoid.
        maxswap: Number of rewirings.

    Returns:
        A: Randomized matrix.

    Notes:
        1. Based on the script randmio_und.m.
        2. Graph may become disconnected as a result of rewiring. Always
           important to check.
        3. A can be weighted, though the weighted degree sequence will not be
           preserved.
    """

    # Find the upper triangular part of the adjacency matrix
    i, j = np.nonzero(np.triu(A, 1))
    m = len(i)
    nswap = 0
    while nswap < maxswap:
        while True:
            # Randomly select two edges
            e1 = np.random.randint(m)
            e2 = np.random.randint(m)
            while e2 == e1:
                e2 = np.random.randint(m)
            a = i[e1]
            b = j[e1]
            c = i[e2]
            d = j[e2]
            # Check if the rewiring is valid
            if (a != c and a != d and b != c and b != d):
                break
        # Randomly decide whether to swap the edges
        if np.random.rand() > 0.5:
            i[e2], j[e2] = d, c
            c, d = i[e2], j[e2]
        # Check if the rewiring is valid and doesn't overlap with B
        if not (A[a, d] or A[c, b] or B[a, d] or B[c, b]):
            # Perform the rewiring
            A[a, d] = A[a, b]
            A[a, b] = 0
            A[d, a] = A[b, a]
            A[b, a] = 0
            A[c, b] = A[c, d]
            A[c, d] = 0
            A[b, c] = A[d, c]
            A[d, c] = 0
            j[e1] = d
            j[e2] = b
            nswap += 1
    return A


