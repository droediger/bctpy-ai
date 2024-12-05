# Translated from clustering_coef_bu.m

import numpy as np

def clustering_coef_bu(G):
    # CLUSTERING_COEF_BU     Clustering coefficient
    #
    #   C = clustering_coef_bu(A);
    #
    #   The clustering coefficient is the fraction of triangles around a node
    #   (equiv. the fraction of node's neighbors that are neighbors of each other).
    #
    #   Input:      A,      binary undirected connection matrix (NumPy array)
    #
    #   Output:     C,      clustering coefficient vector (NumPy array)
    #
    #   Reference: Watts and Strogatz (1998) Nature 393:440-442.
    #
    #
    #   Adapted from Mika Rubinov's MATLAB code, 2007-2010

    n = len(G)
    C = np.zeros((n, 1))

    for u in range(n):
        V = np.where(G[u, :] == 1)[0]  #Find indices where G[u,:] is non-zero
        k = len(V)
        if k >= 2:  # degree must be at least 2
            S = G[np.ix_(V, V)] #Select submatrix
            C[u] = np.sum(S) / (k**2 - k)

    return C.flatten()



