# Translated from latmio_dir_connected.m

import numpy as np

def latmio_dir_connected(R, ITER, D=None):
    """
    Lattice with preserved in/out degree distribution.

    Parameters
    ----------
    R : numpy.ndarray
        Directed (binary/weighted) connection matrix.
    ITER : int
        Rewiring parameter (each edge is rewired approximately ITER times).
    D : numpy.ndarray, optional
        Distance-to-diagonal matrix. If not provided, it will be computed.

    Returns
    -------
    Rlatt : numpy.ndarray
        Latticized network in original node ordering.
    Rrp : numpy.ndarray
        Latticized network in node ordering used for latticization.
    ind_rp : numpy.ndarray
        Node ordering used for latticization.
    eff : int
        Number of actual rewirings carried out.

    References
    ----------
    Maslov and Sneppen (2002) Science 296:910
    Sporns and Zwi (2004) Neuroinformatics 2:145
    """
    n = R.shape[0]

    # Randomly reorder matrix
    ind_rp = np.random.permutation(n)
    R = R[ind_rp][:, ind_rp]

    # Create 'distance to diagonal' matrix
    if D is None:
        D = np.zeros((n, n))
        u = np.concatenate(([0], np.min([np.arange(1, n), np.arange(n - 1, 0, -1)], axis=0)))
        for v in range(1, int(np.ceil(n / 2)) + 1):
            D[n - v, :] = u[v:]
            D[v - 1, :] = D[n - v, ::-1]

    i, j = np.nonzero(R)
    K = len(i)
    ITER = K * ITER

    # Maximal number of rewiring attempts per 'iter'
    maxAttempts = int(np.round(n * K / (n * (n - 1))))
    # Actual number of successful rewirings
    eff = 0

    for iter in range(ITER):
        att = 0
        while att <= maxAttempts:  # while not rewired
            rewire = 1
            while True:
                e1 = int(np.ceil(K * np.random.rand()))
                e2 = int(np.ceil(K * np.random.rand()))
                while e2 == e1:
                    e2 = int(np.ceil(K * np.random.rand()))
                a = i[e1 - 1]
                b = j[e1 - 1]
                c = i[e2 - 1]
                d = j[e2 - 1]

                if (a != c) and (a != d) and (b != c) and (b != d):
                    break  # all four vertices must be different

            # Rewiring condition
            if not (R[a, d] or R[c, b]):
                # Lattice condition
                if (D[a, b] * R[a, b] + D[c, d] * R[c, d]) >= (D[a, d] * R[a, b] + D[c, b] * R[c, d]):
                    # Connectedness condition
                    if not ((R[a, c] or R[d, b] or R[d, c]) and (R[c, a] or R[b, d] or R[b, a])):
                        P = R[[a, c], :]
                        P[0, b] = 0
                        P[0, d] = 1
                        P[1, d] = 0
                        P[1, b] = 1
                        PN = P.copy()
                        PN[0, a] = 1
                        PN[1, c] = 1

                        while True:
                            P[0, :] = np.any(R[np.nonzero(P[0, :])[0], :], axis=0)
                            P[1, :] = np.any(R[np.nonzero(P[1, :])[0], :], axis=0)
                            P = P * (1 - PN)
                            PN = PN + P
                            if not np.all(np.any(P, axis=1)):
                                rewire = 0
                                break
                            elif np.any(PN[0, [b, c]]) and np.any(PN[1, [d, a]]):
                                break

                    if rewire:  # Reassign edges
                        R[a, d] = R[a, b]
                        R[a, b] = 0
                        R[c, b] = R[c, d]
                        R[c, d] = 0

                        j[e1 - 1] = d  # Reassign edge indices
                        j[e2 - 1] = b
                        eff += 1
                        break
            att += 1

    # Lattice in node order used for latticization
    Rrp = R
    # Reverse random permutation of nodes
    ind_rp_reverse = np.argsort(ind_rp)
    Rlatt = Rrp[ind_rp_reverse][:, ind_rp_reverse]
    return Rlatt, Rrp, ind_rp, eff


