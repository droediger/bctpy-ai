# Translated from path_transitivity.m

import numpy as np

def path_transitivity(W, transform=None):
    """
    Transitivity based on shortest paths.

    Computes the density of local detours (triangles) that are available along the shortest-paths between all pairs of nodes.

    Args:
        W (numpy.ndarray): Unweighted/weighted undirected connection weight OR length matrix.
        transform (str, optional): If the input matrix is a connection weight matrix, specify a transform that maps input connection weights to connection lengths.  'log' -> l_ij = -log(w_ij), 'inv' -> l_ij = 1/w_ij. If the input matrix is a connection length matrix, do not specify a transform (or specify an empty transform argument). Defaults to None.

    Returns:
        numpy.ndarray: Matrix of pairwise path transitivity.
    """

    n = len(W)
    m = np.zeros((n, n))
    T = np.zeros((n, n))

    for i in range(n - 1):
        for j in range(i + 1, n):
            x = 0
            y = 0
            z = 0
            for k in range(n):
                if W[i, k] != 0 and W[j, k] != 0 and k != i and k != j:
                    x += W[i, k] + W[j, k]
                if k != j:
                    y += W[i, k]
                if k != i:
                    z += W[j, k]
            if y + z !=0:
                m[i, j] = x / (y + z)

    m = m + m.transpose()

    hops, Pmat = distance_wei_floyd(W, transform) #Assumed to be defined elsewhere

    # --- path transitivity ---
    for i in range(n - 1):
        for j in range(i + 1, n):
            x = 0
            path = retrieve_shortest_path(i, j, hops, Pmat) #Assumed to be defined elsewhere
            K = len(path)
            for t in range(K - 1):
                for l in range(t + 1, K):
                    x += m[path[t], path[l]]
            if K*(K-1) != 0:
                T[i, j] = 2 * x / (K * (K - 1))

    T = T + T.transpose()
    return T

def distance_wei_floyd(W,transform):
    # Dummy function, replace with your actual implementation
    # This is a placeholder.  Replace this with your actual implementation of the Floyd-Warshall algorithm.
    n = len(W)
    hops = np.zeros((n,n))
    Pmat = np.zeros((n,n))
    return hops,Pmat

def retrieve_shortest_path(i,j,hops,Pmat):
    # Dummy function, replace with your actual implementation
    #This is a placeholder. Replace this with your actual implementation of shortest path retrieval.
    return [i,j]

