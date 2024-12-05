# Translated from randmio_und_connected.m

import numpy as np

def randmio_und_connected(R, ITER):
    """Random graph with preserved degree distribution.

    Args:
        R (numpy.ndarray): Undirected (binary/weighted) connection matrix.
        ITER (float): Rewiring parameter (each edge is rewired approximately ITER times).

    Returns:
        tuple: A tuple containing:
            - R (numpy.ndarray): Randomized network.
            - eff (int): Number of actual rewirings carried out.
    """
    n = R.shape[0]
    i, j = np.nonzero(np.tril(R))
    K = len(i)
    ITER = int(K * ITER)

    # Maximal number of rewiring attempts per 'iter'
    maxAttempts = int(np.round(n * K / (n * (n - 1))))
    # Actual number of successful rewirings
    eff = 0

    for iter in range(ITER):
        att = 0
        while att <= maxAttempts:  # while not rewired
            rewire = 1
            while True:
                e1 = np.ceil(K * np.random.rand()) -1
                e2 = np.ceil(K * np.random.rand()) -1
                while e2 == e1:
                    e2 = np.ceil(K * np.random.rand()) -1
                a = i[int(e1)]
                b = j[int(e1)]
                c = i[int(e2)]
                d = j[int(e2)]

                if np.all(a != np.array([c, d])) and np.all(b != np.array([c, d])):
                    break  # all four vertices must be different

            if np.random.rand() > 0.5:
                i[int(e2)], j[int(e2)] = d, c  # flip edge c-d with 50% probability
                c, d = i[int(e2)], j[int(e2)]  # to explore all potential rewirings

            # rewiring condition
            if not (R[a, d] or R[c, b]):
                # connectedness condition
                if not (R[a, c] or R[b, d]):
                    P = R[[a, d], :]
                    P[0, b] = 0
                    P[1, c] = 0
                    PN = P
                    PN[:, d] = 1
                    PN[:, a] = 1

                    while True:
                        P[0, :] = np.any(R[np.nonzero(P[0, :])[0], :], axis=0)
                        P[1, :] = np.any(R[np.nonzero(P[1, :])[0], :], axis=0)
                        P = P * (1 - PN)
                        if not np.all(np.any(P, axis=1)):
                            rewire = 0
                            break
                        elif np.any(P[:, [b, c]]):
                            break
                        PN = PN + P

                # connectedness testing
                if rewire:  # reassign edges
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

