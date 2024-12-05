# Translated from module_degree_zscore.m

import numpy as np

def module_degree_zscore(W, Ci, flag=0):
    """Within-module degree z-score

    The within-module degree z-score is a within-module version of degree centrality.

    Args:
        W: binary/weighted, directed/undirected connection matrix
        Ci: community affiliation vector
        flag: 0, undirected graph (default)
              1, directed graph: out-degree
              2, directed graph: in-degree
              3, directed graph: out-degree and in-degree

    Returns:
        Z: within-module degree z-score.

    Reference: Guimera R, Amaral L. Nature (2005) 433:895-900.
    """

    if flag == 2:
        W = W.T  # For in-degree, transpose the matrix
    elif flag == 3:
        W = W + W.T  # For both in and out-degree, symmetrize the matrix

    n = len(W)  # Number of vertices
    Z = np.zeros((n, 1))

    for i in np.unique(Ci):
        Koi = np.sum(W[Ci == i, :][:, Ci == i], axis=1) # Sum of connection matrix for each module
        
        # Avoid division by zero if standard deviation is zero.
        if np.std(Koi) ==0:
            Z[Ci == i] = 0
        else:
            Z[Ci == i] = (Koi - np.mean(Koi)) / np.std(Koi)

    Z[np.isnan(Z)] = 0  #Handle NaN values, which can occur if a module has only one node

    return Z.flatten() #Return as a 1D array



