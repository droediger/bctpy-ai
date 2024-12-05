# Translated from pagerank_centrality.m

import numpy as np

def pagerank_centrality(A, d, falff=None):
    """PageRank centrality

    Args:
        A (numpy.ndarray): Adjacency matrix.
        d (float): Damping factor (typically 0.85).
        falff (numpy.ndarray, optional): Initial PageRank probability. Defaults to None.

    Returns:
        numpy.ndarray: Vector of PageRank scores.

    Notes:
        The algorithm works well for smaller matrices (number of nodes around 1000 or less).
        References:
            [1]. GeneRank: Using search engine technology for the analysis of microarray experiments, by Julie L. Morrison, Rainer Breitling, Desmond J. Higham and David R. Gilbert, BMC Bioinformatics, 6:233, 2005.
            [2]. Boldi P, Santini M, Vigna S (2009) PageRank: Functional dependencies. ACM Trans Inf Syst 27, 1-23.
    """
    N = A.shape[0]
    if falff is None:
        norm_falff = np.ones((N, 1)) / N
    else:
        falff = np.abs(falff)
        norm_falff = falff / np.sum(falff)

    deg = np.sum(A, axis=1)
    ind = (deg == 0)
    deg[ind] = 1
    D1 = np.diag(1. / deg)
    B = np.eye(N) - d * (A @ D1)
    b = (1 - d) * norm_falff
    r = np.linalg.solve(B, b)
    r = r / np.sum(r)
    return r



