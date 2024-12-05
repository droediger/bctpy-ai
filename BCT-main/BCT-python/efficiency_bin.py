# Translated from efficiency_bin.m

import numpy as np

def efficiency_bin(A, local=0):
    """
    Global efficiency, local efficiency.

    Eglob = efficiency_bin(A)
    Eloc = efficiency_bin(A, 1)

    The global efficiency is the average of inverse shortest path length,
    and is inversely related to the characteristic path length.

    The local efficiency is the global efficiency computed on the
    neighborhood of the node, and is related to the clustering coefficient.

    Args:
        A (numpy.ndarray): Binary undirected or directed connection matrix.
        local (int, optional): Optional argument. 
                               local=0 computes global efficiency (default).
                               local=1 computes local efficiency.

    Returns:
        numpy.ndarray: Eglob, global efficiency (scalar) or 
                       Eloc, local efficiency (vector).

    Algorithm: algebraic path count

    References:
        Latora and Marchiori (2001) Phys Rev Lett 87:198701.
        Fagiolo (2007) Phys Rev E 76:026107.
        Rubinov M, Sporns O (2010) NeuroImage 52:1059-69

    """
    n = len(A)  # Number of nodes
    A[np.arange(n), np.arange(n)] = 0  # Clear diagonal
    A = A.astype(float)  # Enforce double precision

    if local:  # Local efficiency
        E = np.zeros((n, 1))
        for u in range(n):
            V = np.where((A[u, :] + A[:, u].T) > 0)[0]  # Neighbors
            sa = A[u, V] + A[V, u].T  # Symmetrized adjacency vector
            e = distance_inv(A[np.ix_(V, V)])  # Inverse distance matrix
            se = e + e.T  # Symmetrized inverse distance matrix
            numer = np.sum((sa.T @ sa) * se) / 2  # Numerator
            if numer != 0:
                denom = np.sum(sa) ** 2 - np.sum(sa ** 2)  # Denominator
                E[u] = numer / denom  # Local efficiency
    else:  # Global efficiency
        e = distance_inv(A)
        E = np.sum(e) / (n ** 2 - n)

    return E


def distance_inv(A_):
    l = 1  # Path length
    Lpath = A_.copy()  # Matrix of paths l
    D = A_.copy()  # Distance matrix
    n_ = len(A_)

    Idx = np.ones((n_, n_), dtype=bool)
    while np.any(Idx):
        l += 1
        Lpath = Lpath @ A_
        Idx = (Lpath != 0) & (D == 0)
        D[Idx] = l

    D[~D | np.eye(n_, dtype=bool)] = np.inf  # Assign inf to disconnected nodes and to diagonal
    D = 1. / D  # Invert distance
    return D


