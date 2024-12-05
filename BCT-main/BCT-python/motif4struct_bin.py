# Translated from motif4struct_bin.m

import numpy as np

def motif4struct_bin(A):
    """
    Frequency of structural class-4 motifs

    Parameters
    ----------
    A : numpy.ndarray
        Binary directed connection matrix

    Returns
    -------
    F : numpy.ndarray
        Node motif frequency fingerprint
    f : numpy.ndarray
        Network motif frequency fingerprint

    Notes
    -----
    The function find_motif34.m (not provided) outputs the motif legend.

    References
    ----------
    Milo et al. (2002) Science 298:824-827
    Sporns O, Kötter R (2004) PLoS Biol 2: e369
    """

    M4n = np.load('motif34lib.npz')['M4n']  # Load motif data
    ID4 = np.load('motif34lib.npz')['ID4']

    n = len(A)  # Number of vertices in A
    F = np.zeros((199, n))  # Motif count of each vertex
    f = np.zeros((199, 1))  # Motif count for whole graph
    As = A | A.T  # Symmetric adjacency matrix

    for u in range(n - 3):  # Loop u 1:n-2
        V1 = np.concatenate((np.zeros(u, dtype=bool), As[u, u + 1:]))  # v1: neibs of u (>u)
        for v1 in np.where(V1)[0]:
            V2 = np.concatenate((np.zeros(u, dtype=bool), As[v1, u + 1:]))  # v2: all neibs of v1 (>u)
            V2[V1] = False  # not already in V1
            V2 = V2 | np.concatenate((np.zeros(v1, dtype=bool), As[u, v1 + 1:]))  # and all neibs of u (>v1)
            for v2 in np.where(V2)[0]:
                vz = max(v1, v2)  # vz: largest rank node
                V3 = np.concatenate((np.zeros(u, dtype=bool), As[v2, u + 1:]))  # v3: all neibs of v2 (>u)
                V3[V2] = False  # not already in V1&V2
                V3 = V3 | np.concatenate((np.zeros(v2, dtype=bool), As[v1, v2 + 1:]))  # and all neibs of v1 (>v2)
                V3[V1] = False  # not already in V1
                V3 = V3 | np.concatenate((np.zeros(vz, dtype=bool), As[u, vz + 1:]))  # and all neibs of u (>vz)
                for v3 in np.where(V3)[0]:
                    s = np.sum(10**(np.arange(11)[::-1]) * np.array([A[v1, u], A[v2, u], A[v3, u],
                                                                       A[u, v1], A[v2, v1], A[v3, v1],
                                                                       A[u, v2], A[v1, v2],
                                                                       A[v3, v2], A[u, v3], A[v1, v3], A[v2, v3]]).astype(np.uint64))
                    ind = np.where(s == M4n)[0]
                    if 2 == 2:  #Replaces nargout==2 which is not directly translatable.  Assumes it should always execute inner conditional.
                        F[ind, [u, v1, v2, v3]] += 1
                    f[ind] += 1
    return f, F


