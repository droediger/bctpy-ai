# Translated from latmio_und.m

import numpy as np

def latmio_und(R, ITER, D=None):
    """
    Lattice with preserved degree distribution.

    Parameters
    ----------
    R : numpy.ndarray
        Undirected (binary/weighted) connection matrix.
    ITER : float
        Rewiring parameter (each edge is rewired approximately ITER times).
    D : numpy.ndarray, optional
        Distance-to-diagonal matrix. If not provided, it will be calculated.

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
    R = R[ind_rp, :][:, ind_rp]

    # Create 'distance to diagonal' matrix
    if D is None:
        D = np.zeros((n, n))
        u = np.concatenate(([0], np.min([np.arange(1, n), np.arange(n - 1, 0, -1)], axis=0)))
        for v in range(1, int(np.ceil(n / 2)) + 1):
            D[n - v, :] = u[v:n + 1]
            D[v - 1, :] = D[n - v, :][::-1]

    # Find lower triangular indices
    i, j = np.nonzero(np.tril(R))
    K = len(i)
    ITER = int(K * ITER)

    # Maximal number of rewiring attempts per 'iter'
    maxAttempts = int(np.round(n * K / (n * (n - 1) / 2)))
    # Actual number of successful rewirings
    eff = 0

    for iter in range(ITER):
        att = 0
        while att <= maxAttempts:  # While not rewired
            while True:
                e1 = np.ceil(K * np.random.rand()).astype(int)
                e2 = np.ceil(K * np.random.rand()).astype(int)
                while e2 == e1:
                    e2 = np.ceil(K * np.random.rand()).astype(int)
                a = i[e1 -1]
                b = j[e1 -1]
                c = i[e2 -1]
                d = j[e2 -1]

                if (a != c) and (a != d) and (b != c) and (b != d):
                    break  # All four vertices must be different

            if np.random.rand() > 0.5:
                i[e2 - 1], j[e2 - 1] = d, c  # Flip edge c-d with 50% probability
                c, d = i[e2-1], j[e2-1]  # To explore all potential rewirings

            # Rewiring condition
            if (R[a, d] == 0) and (R[c, b] == 0):
                # Lattice condition
                if (D[a, b] * R[a, b] + D[c, d] * R[c, d]) >= (D[a, d] * R[a, b] + D[c, b] * R[c, d]):
                    R[a, d] = R[a, b]
                    R[a, b] = 0
                    R[d, a] = R[b, a]
                    R[b, a] = 0
                    R[c, b] = R[c, d]
                    R[c, d] = 0
                    R[b, c] = R[d, c]
                    R[d, c] = 0

                    j[e1 - 1] = d  # Reassign edge indices
                    j[e2 - 1] = b
                    eff += 1
                    break
            att += 1

    # Lattice in node order used for latticization
    Rrp = R
    # Reverse random permutation of nodes
    ind_rp_reverse = np.argsort(ind_rp)
    Rlatt = Rrp[ind_rp_reverse, :][:, ind_rp_reverse]

    return Rlatt, Rrp, ind_rp, eff


