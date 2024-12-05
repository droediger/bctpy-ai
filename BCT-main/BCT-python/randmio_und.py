# Translated from randmio_und.m

import numpy as np

def randmio_und(R, ITER):
    """
    Random graph with preserved degree distribution.

    Parameters
    ----------
    R : numpy.ndarray
        Undirected (binary/weighted) connection matrix.
    ITER : int
        Rewiring parameter (each edge is rewired approximately ITER times).

    Returns
    -------
    R : numpy.ndarray
        Randomized network.
    eff : int
        Number of actual rewirings carried out.

    References
    ----------
    Maslov and Sneppen (2002) Science 296:910
    """

    n = R.shape[0]  #number of nodes
    i, j = np.nonzero(np.tril(R)) #indices of lower triangular part of R
    K = len(i)  #number of edges
    ITER = K * ITER  #total number of rewiring attempts

    # Maximal number of rewiring attempts per 'iter'
    maxAttempts = np.round(n * K / (n * (n - 1)))
    # Actual number of successful rewirings
    eff = 0

    for iter in range(ITER):
        att = 0
        while att <= maxAttempts:  # while not rewired
            while True:
                e1 = np.ceil(K * np.random.rand()) -1 #adjust for 0-based indexing
                e2 = np.ceil(K * np.random.rand()) -1 #adjust for 0-based indexing
                while e2 == e1:
                    e2 = np.ceil(K * np.random.rand()) -1 #adjust for 0-based indexing
                a = i[int(e1)]
                b = j[int(e1)]
                c = i[int(e2)]
                d = j[int(e2)]

                if (a != c) and (a != d) and (b != c) and (b != d):
                    break  # all four vertices must be different
            
            if np.random.rand() > 0.5:
                i[int(e2)], j[int(e2)] = d, c  # flip edge c-d with 50% probability
                c, d = i[int(e2)], j[int(e2)]  # to explore all potential rewirings

            # Rewiring condition
            if not (R[a, d] or R[c, b]):
                R[a, d] = R[a, b]
                R[a, b] = 0
                R[d, a] = R[b, a]
                R[b, a] = 0
                R[c, b] = R[c, d]
                R[c, d] = 0
                R[b, c] = R[d, c]
                R[d, c] = 0

                j[int(e1)] = d  # reassign edge indices
                j[int(e2)] = b
                eff += 1
                break
            att += 1
    return R, eff

