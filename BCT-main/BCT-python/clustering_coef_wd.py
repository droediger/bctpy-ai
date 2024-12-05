# Translated from clustering_coef_wd.m

import numpy as np

def clustering_coef_wd(W):
    """
    Clustering coefficient.

    The weighted clustering coefficient is the average "intensity"
    (geometric mean) of all triangles associated with each node.

    Args:
        W (numpy.ndarray): Weighted directed connection matrix (all weights must be between 0 and 1).

    Returns:
        numpy.ndarray: Clustering coefficient vector.
    """

    # Adjacency matrix
    A = W != 0
    # Symmetrized weights matrix ^1/3
    S = W**(1/3) + W.transpose()**(1/3)
    # Total degree (in + out)
    K = np.sum(A + A.transpose(), axis=1)
    # Number of 3-cycles (i.e., directed triangles)
    cyc3 = np.diag(np.linalg.matrix_power(S, 3)) / 2
    # If no 3-cycles exist, make C=0 (via K=inf)
    K[cyc3 == 0] = np.inf
    # Number of all possible 3-cycles
    CYC3 = K * (K - 1) - 2 * np.diag(np.linalg.matrix_power(A, 2))
    # Clustering coefficient
    C = cyc3 / CYC3
    return C


