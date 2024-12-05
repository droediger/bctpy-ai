# Translated from latmio_und_connected.m

import numpy as np

def latmio_und_connected(R, ITER, D=None):
    """
    Lattice with preserved degree distribution.

    This function "latticizes" an undirected network, while preserving the 
    degree distribution. The function does not preserve the strength 
    distribution in weighted networks. The function also ensures that the 
    randomized network maintains connectedness, the ability for every node 
    to reach every other node in the network. The input network for this 
    function must be connected.

    Parameters
    ----------
    R : numpy.ndarray
        Undirected (binary/weighted) connection matrix.
    ITER : float
        Rewiring parameter (each edge is rewired approximately ITER times).
    D : numpy.ndarray, optional
        Distance-to-diagonal matrix. If not provided, it is calculated.

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
    """

    n = R.shape[0]

    # Randomly reorder matrix
    ind_rp = np.random.permutation(n)
    R = R[ind_rp][:, ind_rp]

    # Create 'distance to diagonal' matrix
    if D is None:
        D = np.zeros((n, n))
        u = np.concatenate(([0], np.minimum(np.arange(1, n), np.arange(n - 1, 0, -1))))
        for v in range(1, int(np.ceil(n / 2)) + 1):
            D[n - v, :] = u[v:]
            D[v -1, :] = D[n - v, ::-1]

    i, j = np.nonzero(np.tril(R))
    K = len(i)
    ITER = int(K * ITER)

    # Maximal number of rewiring attempts per 'iter'
    maxAttempts = int(np.round(n * K / (n * (n - 1) / 2)))
    # Actual number of successful rewirings
    eff = 0

    for iter in range(ITER):
        att = 0
        while att <= maxAttempts:  # while not rewired
            rewire = 1
            while True:
                e1 = np.ceil(K * np.random.rand()).astype(int) -1
                e2 = np.ceil(K * np.random.rand()).astype(int) -1
                while e2 == e1:
                    e2 = np.ceil(K * np.random.rand()).astype(int) -1
                a = i[e1]
                b = j[e1]
                c = i[e2]
                d = j[e2]

                if (a != c) and (a != d) and (b != c) and (b != d):
                    break  # all four vertices must be different

            if np.random.rand() > 0.5:
                i[e2], j[e2] = d, c  # flip edge c-d with 50% probability
                c, d = i[e2], j[e2]  # to explore all potential rewirings

            # Rewiring condition
            if (R[a, d] == 0) and (R[c, b] == 0):
                # Lattice condition
                if (D[a, b] * R[a, b] + D[c, d] * R[c, d]) >= (D[a, d] * R[a, b] + D[c, b] * R[c, d]):
                    # Connectedness condition
                    if (R[a, c] == 0) and (R[b, d] == 0):
                        P = np.copy(R[[a, d], :])
                        P[0, b] = 0
                        P[1, c] = 0
                        PN = np.copy(P)
                        PN[:, d] = 1
                        PN[:, a] = 1

                        while True:
                            P[0, :] = np.any(R[P[0, :] != 0, :], axis=0)
                            P[1, :] = np.any(R[P[1, :] != 0, :], axis=0)
                            P = P * (1 - PN)
                            if not np.all(np.any(P, axis=1)):
                                rewire = 0
                                break
                            elif np.any(P[:, [b, c]]):
                                break
                            PN += P

                    if rewire:  # reassign edges
                        R[a, d] = R[a, b]
                        R[a, b] = 0
                        R[d, a] = R[b, a]
                        R[b, a] = 0
                        R[c, b] = R[c, d]
                        R[c, d] = 0
                        R[b, c] = R[d, c]
                        R[d, c] = 0

                        j[e1] = d  # reassign edge indices
                        j[e2] = b
                        eff += 1
                        break
            att += 1

    # Lattice in node order used for latticization
    Rrp = R
    # Reverse random permutation of nodes
    ind_rp_reverse = np.argsort(ind_rp)
    Rlatt = Rrp[ind_rp_reverse][:, ind_rp_reverse]

    return Rlatt, Rrp, ind_rp, eff

