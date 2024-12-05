# Translated from kcore_bd.m

import numpy as np

def kcore_bd(CIJ, k):
    """
    K-core decomposition of a directed graph.

    Args:
        CIJ: Connection/adjacency matrix (binary, directed).
        k: Level of k-core.

    Returns:
        CIJkcore: Connection matrix of the k-core.  Contains only nodes of degree at least k.
        kn: Size of k-core.
        peelorder: Indices of nodes in the order they were peeled away.
        peellevel: Corresponding level at which nodes were peeled away.  Nodes at the same level were peeled at the same time.
    """

    peelorder = np.array([], dtype=int).reshape(0,1) # Initialize as empty array with correct data type
    peellevel = np.array([], dtype=int).reshape(0,1) # Initialize as empty array with correct data type
    iter = 0

    while True:
        # Get degrees of matrix
        id, od, deg = degrees_dir(CIJ) # Assume degrees_dir is defined elsewhere

        # Find nodes with degree < k
        ff = np.where((deg < k) & (deg > 0))[0]

        # If none found -> stop
        if len(ff) == 0:
            break

        # Peel away found nodes
        iter += 1
        CIJ[ff, :] = 0
        CIJ[:, ff] = 0

        peelorder = np.concatenate((peelorder, ff[:,None]), axis=0)
        peellevel = np.concatenate((peellevel, np.full((len(ff), 1), iter)), axis=0)

    CIJkcore = CIJ
    kn = np.sum(deg > 0)
    return CIJkcore, kn, peelorder, peellevel

# Placeholder for degrees_dir function.  Replace with your actual implementation.
def degrees_dir(CIJ):
    id = np.sum(CIJ, axis=0)
    od = np.sum(CIJ, axis=1)
    deg = id + od
    return id, od, deg


