# Translated from randmio_dir.m

import numpy as np

def randmio_dir(R, ITER):
    """Random graph with preserved in/out degree distribution.

    Args:
        R: directed (binary/weighted) connection matrix.
        ITER: rewiring parameter (each edge is rewired approximately ITER times).

    Returns:
        R: randomized network.
        eff: number of actual rewirings carried out.
    """

    n = R.shape[0]  #number of nodes
    i, j = np.nonzero(R)  #indices of non-zero elements
    K = len(i)  #number of edges
    ITER = K * ITER  #total number of rewiring attempts

    #maximal number of rewiring attempts per 'iter'
    maxAttempts = int(np.round(n * K / (n * (n - 1))))
    #actual number of successful rewirings
    eff = 0

    for iter in range(ITER):
        att = 0
        while att <= maxAttempts:  # while not rewired
            while True:
                e1 = np.ceil(K * np.random.rand()).astype(int)
                e2 = np.ceil(K * np.random.rand()).astype(int)
                while e2 == e1:
                    e2 = np.ceil(K * np.random.rand()).astype(int)
                a = i[e1 - 1]
                b = j[e1 - 1]
                c = i[e2 - 1]
                d = j[e2 - 1]

                if (a != c) and (a != d) and (b != c) and (b != d):
                    break  # all four vertices must be different

            #rewiring condition
            if not (R[a, d] or R[c, b]):
                R[a, d] = R[a, b]
                R[a, b] = 0
                R[c, b] = R[c, d]
                R[c, d] = 0

                j[e1 - 1] = d  # reassign edge indices
                j[e2 - 1] = b
                eff += 1
                break
            att += 1
    return R, eff

