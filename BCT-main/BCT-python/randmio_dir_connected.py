# Translated from randmio_dir_connected.m

import numpy as np

def randmio_dir_connected(R, ITER):
    """Random graph with preserved in/out degree distribution.

    Args:
        R: directed (binary/weighted) connection matrix.
        ITER: rewiring parameter (each edge is rewired approximately ITER times).

    Returns:
        R: randomized network.
        eff: number of actual rewirings carried out.
    """

    n = R.shape[0]  # number of nodes
    i, j = np.nonzero(R)  # indices of non-zero elements (edges)
    K = len(i)  # number of edges
    ITER = K * ITER  # total number of rewiring attempts

    # Maximal number of rewiring attempts per 'iter'
    maxAttempts = np.round(n * K / (n * (n - 1)))
    # Actual number of successful rewirings
    eff = 0

    for iter in range(ITER):
        att = 0
        while att <= maxAttempts:  # while not rewired
            rewire = 1
            while True:
                e1 = np.ceil(K * np.random.rand()).astype(int) -1 #adjust for 0-based indexing
                e2 = np.ceil(K * np.random.rand()).astype(int) -1 #adjust for 0-based indexing
                while e2 == e1:
                    e2 = np.ceil(K * np.random.rand()).astype(int) -1 #adjust for 0-based indexing
                a = i[e1]
                b = j[e1]
                c = i[e2]
                d = j[e2]

                if (a != c) and (a != d) and (b != c) and (b != d):
                    break  # all four vertices must be different

            # Rewiring condition
            if not (R[a, d] or R[c, b]):
                # Connectedness condition
                if not ((np.any(np.array([R[a,c], R[d,b], R[d,c]])) and np.any(np.array([R[c,a], R[b,d], R[b,a]])))):
                    P = R[[a, c], :]
                    P[0, b] = 0
                    P[0, d] = 1
                    P[1, d] = 0
                    P[1, b] = 1
                    PN = P.copy()
                    PN[0, a] = 1
                    PN[1, c] = 1

                    while True:
                        P[0, :] = np.any(R[P[0, :] != 0, :], axis=0)
                        P[1, :] = np.any(R[P[1, :] != 0, :], axis=0)
                        P = P * (PN == 0)
                        PN = PN + P
                        if not np.all(np.any(P, axis=1)):
                            rewire = 0
                            break
                        elif np.any(PN[0, [b, c]]) and np.any(PN[1, [d, a]]):
                            break

            # edge reassignment
            if rewire:
                R[a, d] = R[a, b]
                R[a, b] = 0
                R[c, b] = R[c, d]
                R[c, d] = 0

                j[e1] = d  # reassign edge indices
                j[e2] = b
                eff += 1
                break
            att += 1

    return R, eff

