# Translated from motif3struct_bin.m

import numpy as np

def motif3struct_bin(A):
    """
    Frequency of structural class-3 motifs

    Parameters
    ----------
    A : numpy.ndarray
        Binary directed connection matrix

    Returns
    -------
    f : numpy.ndarray
        Network motif frequency fingerprint
    F : numpy.ndarray
        Node motif frequency fingerprint

    Notes
    -----
    The function find_motif34.m (not included here) outputs the motif legend.

    References
    ----------
    Milo et al. (2002) Science 298:824-827
    Sporns O, Kötter R (2004) PLoS Biol 2: e369

    """
    M3n, ID3 = load_motif_data() # Load motif data; assumes load_motif_data is defined elsewhere

    n = len(A)  # Number of vertices in A
    F = np.zeros((13, n))  # Motif count of each vertex
    f = np.zeros((13, 1))  # Motif count for whole graph
    As = A | A.T  # Symmetrized adjacency matrix

    for u in range(n - 2):  # Loop u 1:n-2
        V1 = np.concatenate((np.zeros(u, dtype=bool), As[u, u+1:]))  # v1: neibs of u (>u)
        for v1_index in np.nonzero(V1)[0]:
            v1 = v1_index + u +1
            V2 = np.concatenate((np.zeros(u, dtype=bool), As[v1 -1, u+1:]))  # v2: all neibs of v1 (>u)
            V2[V1] = 0  # not already in V1
            V2 = np.logical_or(np.concatenate((np.zeros(v1 -1, dtype=bool), As[u, v1:])), V2)  # and all neibs of u (>v1)
            for v2_index in np.nonzero(V2)[0]:
                v2 = v2_index + u + 1
                s = np.sum(10**(5 - np.arange(6)) * np.array([A[v1 - 1, u], A[v2 - 1, u], A[u, v1 - 1],
                                                              A[v2 - 1, v1 - 1], A[u, v2 - 1], A[v1 - 1, v2 - 1]]).astype(int))
                ind = np.nonzero(ID3 == s)[0]
                if len(ind) > 0: #check that ind is not empty
                    if F.shape[1] > 0:
                        F[ind, [u, v1_index + u, v2_index + u]] += 1
                    f[ind] +=1
    return f, F


def load_motif_data():
    # Replace this with your actual data loading logic.  This is a placeholder.
    #  This function should load M3n and ID3 from a file or other source.
    M3n = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]) # Example data
    ID3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]) # Example data
    return M3n, ID3

