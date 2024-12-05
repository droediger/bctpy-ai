# Translated from motif4struct_wei.m

import numpy as np

def motif4struct_wei(W):
    """
    Intensity and coherence of structural class-4 motifs.

    Calculates the motif intensity, coherence, and frequency for each node in a weighted directed network.

    Args:
        W: Weighted directed connection matrix (all weights must be between 0 and 1).

    Returns:
        I: Node motif intensity fingerprint.
        Q: Node motif coherence fingerprint.
        F: Node motif frequency fingerprint.
    """

    M4, M4n, ID4, N4 = _load_motif_data()  # Load motif data

    n = len(W)  # Number of vertices in W
    I = np.zeros((199, n))  # Intensity
    Q = np.zeros((199, n))  # Coherence
    F = np.zeros((199, n))  # Frequency

    A = (W != 0).astype(int)  # Adjacency matrix
    As = A | A.T  # Symmetrized adjacency matrix

    for u in range(n - 3):  # Loop u 1:n-2
        V1 = np.concatenate((np.zeros(u, dtype=bool), As[u, u + 1:]))  # v1: neighbors of u (>u)
        for v1_index in np.where(V1)[0]:
            v1 = v1_index + u +1
            V2 = np.concatenate((np.zeros(u, dtype=bool), As[v1 - 1, u + 1:]))  # v2: all neighbors of v1 (>u)
            V2[V1] = 0  # Not already in V1
            V2 = V2 | np.concatenate((np.zeros(v1 -1, dtype=bool), As[u, v1:]))  # and all neighbors of u (>v1)
            for v2_index in np.where(V2)[0]:
                v2 = v2_index + u + 1
                vz = max(v1, v2)  # vz: largest rank node
                V3 = np.concatenate((np.zeros(u, dtype=bool), As[v2 - 1, u + 1:]))  # v3: all neighbors of v2 (>u)
                V3[V2] = 0  # Not already in V1&V2
                V3 = V3 | np.concatenate((np.zeros(v2 - 1, dtype=bool), As[v1 - 1, v2:]))  # and all neighbors of v1 (>v2)
                V3[V1] = 0  # Not already in V1
                V3 = V3 | np.concatenate((np.zeros(vz - 1, dtype=bool), As[u, vz:]))  # and all neighbors of u (>vz)
                for v3_index in np.where(V3)[0]:
                    v3 = v3_index + u + 1
                    w = np.array([W[v1 - 1, u], W[v2 - 1, u], W[v3 - 1, u], W[u, v1 - 1], W[v2 - 1, v1 - 1],
                                  W[v3 - 1, v1 - 1], W[u, v2 - 1], W[v1 - 1, v2 - 1], W[v3 - 1, v2 - 1],
                                  W[u, v3 - 1], W[v1 - 1, v3 - 1], W[v2 - 1, v3 - 1]])
                    s = np.sum(10**np.arange(11, -1, -1) * np.array([A[v1 - 1, u], A[v2 - 1, u], A[v3 - 1, u],
                                                                     A[u, v1 - 1], A[v2 - 1, v1 - 1],
                                                                     A[v3 - 1, v1 - 1], A[u, v2 - 1],
                                                                     A[v1 - 1, v2 - 1], A[v3 - 1, v2 - 1],
                                                                     A[u, v3 - 1], A[v1 - 1, v3 - 1],
                                                                     A[v2 - 1, v3 - 1]]))
                    ind = (s == M4n)

                    M = w * M4[ind, :]
                    id = ID4[ind]
                    l = N4[ind]
                    x = np.sum(M, axis=1) / l  # Arithmetic mean
                    M[M == 0] = 1  # Enable geometric mean
                    i = np.prod(M, axis=1)**(1 / l)  # Intensity
                    q = i / x  # Coherence

                    # Add to cumulative count
                    I[id, [u, v1 - 1, v2 - 1, v3 - 1]] += [i[0],i[0],i[0],i[0]]
                    Q[id, [u, v1 - 1, v2 - 1, v3 - 1]] += [q[0],q[0],q[0],q[0]]
                    F[id, [u, v1 - 1, v2 - 1, v3 - 1]] += [1, 1, 1, 1]

    return I, Q, F


def _load_motif_data():
    #Simulate loading data - replace with actual loading if needed.
    M4 = np.random.rand(199,12)
    M4n = np.random.randint(0,1000,199)
    ID4 = np.random.randint(0,199,199)
    N4 = np.random.randint(1,13,199)

    return M4, M4n, ID4, N4


