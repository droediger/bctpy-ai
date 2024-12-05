# Translated from motif4funct_wei.m

import numpy as np

def motif4funct_wei(W):
    """
    Intensity and coherence of functional class-4 motifs

    Parameters
    ----------
    W : numpy.ndarray
        Weighted directed connection matrix (all weights must be between 0 and 1)

    Returns
    -------
    I : numpy.ndarray
        Node motif intensity fingerprint
    Q : numpy.ndarray
        Node motif coherence fingerprint
    F : numpy.ndarray
        Node motif frequency fingerprint

    Notes
    -----
    1. The function find_motif34.m (assumed to be defined elsewhere) outputs the motif legend.
    2. Average intensity and coherence are given by I./F and Q./F
    3. All weights must be between 0 and 1.  This may be achieved using a weight normalization function (assumed to be defined elsewhere).
    4. There is a source of possible confusion in motif terminology. Motifs ("structural" and "functional") are most frequently considered only in the context of anatomical brain networks (Sporns and Kötter, 2004). On the other hand, motifs are not commonly studied in undirected networks, due to the paucity of local undirected connectivity patterns.

    References
    ----------
    Onnela et al. (2005), Phys Rev E 71:065103
    Milo et al. (2002) Science 298:824-827
    Sporns O, Kötter R (2004) PLoS Biol 2: e369
    """

    # Load motif data (assuming motif34lib.mat exists and contains M4, ID4, N4)
    try:
        M4, ID4, N4 = np.load('motif34lib.npz')['arr_0'], np.load('motif34lib.npz')['arr_1'], np.load('motif34lib.npz')['arr_2']
    except FileNotFoundError:
        raise FileNotFoundError("motif34lib.npz not found. Please provide the motif data.")


    n = len(W)  # Number of vertices in W
    I = np.zeros((199, n))  # Intensity
    Q = np.zeros((199, n))  # Coherence
    F = np.zeros((199, n))  # Frequency

    A = (W != 0).astype(int)  # Adjacency matrix
    As = A | A.T  # Symmetrized adjacency

    for u in range(n - 3):  # Loop u 1:n-2
        V1 = np.concatenate((np.zeros(u, dtype=bool), As[u, u + 1:]))  # v1: neibs of u (>u)
        for v1_index in np.where(V1)[0]:
            v1 = v1_index + u + 1
            V2 = np.concatenate((np.zeros(u, dtype=bool), As[v1 -1, u + 1:]))  # v2: all neibs of v1 (>u)
            V2[V1] = False  # not already in V1
            V2 = V2 | np.concatenate((np.zeros(v1, dtype=bool), As[u, v1:]))  # and all neibs of u (>v1)
            for v2_index in np.where(V2)[0]:
                v2 = v2_index + u + 1
                vz = max(v1, v2)  # vz: largest rank node
                V3 = np.concatenate((np.zeros(u, dtype=bool), As[v2 - 1, u + 1:]))  # v3: all neibs of v2 (>u)
                V3[V2] = False  # not already in V1&V2
                V3 = V3 | np.concatenate((np.zeros(v2, dtype=bool), As[v1 - 1, v2:]))  # and all neibs of v1 (>v2)
                V3[V1] = False  # not already in V1
                V3 = V3 | np.concatenate((np.zeros(vz, dtype=bool), As[u, vz:]))  # and all neibs of u (>vz)
                for v3_index in np.where(V3)[0]:
                    v3 = v3_index + u + 1

                    w = np.array([W[v1 - 1, u], W[v2 - 1, u], W[v3 - 1, u], W[u, v1 - 1], W[v2 - 1, v1 - 1], W[v3 - 1, v1 - 1],
                                  W[u, v2 - 1], W[v1 - 1, v2 - 1], W[v3 - 1, v2 - 1], W[u, v3 - 1], W[v1 - 1, v3 - 1], W[v2 - 1, v3 - 1]])
                    a = np.array([A[v1 - 1, u], A[v2 - 1, u], A[v3 - 1, u], A[u, v1 - 1], A[v2 - 1, v1 - 1], A[v3 - 1, v1 - 1],
                                  A[u, v2 - 1], A[v1 - 1, v2 - 1], A[v3 - 1, v2 - 1], A[u, v3 - 1], A[v1 - 1, v3 - 1], A[v2 - 1, v3 - 1]])
                    ind = (M4 @ a == N4)  # find all contained isomorphs
                    m = np.sum(ind)  # number of isomorphs

                    M = M4[ind, :] * np.tile(w, (m, 1))
                    id = ID4[ind]
                    l = N4[ind]
                    x = np.sum(M, axis=1) / l  # arithmetic mean
                    M[M == 0] = 1  # enable geometric mean
                    i = np.prod(M, axis=1) ** (1. / l)  # intensity
                    q = i / x  # coherence

                    idu, j = np.unique(id, return_index=True)  # unique motif occurences
                    j = np.concatenate(([0], j))
                    mu = len(idu)  # number of unique motifs
                    i2 = np.zeros(mu)
                    q2 = np.zeros(mu)
                    f2 = np.zeros(mu)

                    for h in range(mu):  # for each unique motif
                        i2[h] = np.sum(i[j[h] + 1:j[h + 1]])  # sum all intensities,
                        q2[h] = np.sum(q[j[h] + 1:j[h + 1]])  # coherences
                        f2[h] = j[h + 1] - j[h]  # and frequencies

                    # then add to cumulative count
                    I[idu, [u, v1 - 1, v2 - 1, v3 - 1]] += np.tile(i2,4)
                    Q[idu, [u, v1 - 1, v2 - 1, v3 - 1]] += np.tile(q2,4)
                    F[idu, [u, v1 - 1, v2 - 1, v3 - 1]] += np.tile(f2,4)
    return I, Q, F


