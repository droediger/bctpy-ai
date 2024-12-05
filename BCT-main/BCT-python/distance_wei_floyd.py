# Translated from distance_wei_floyd.m

import numpy as np

def distance_wei_floyd(D, transform=None):
    """
    Distance matrix (Floyd-Warshall algorithm)

    Computes the topological length of the shortest possible path connecting every pair of nodes in the network.

    Args:
        D (numpy.ndarray): Weighted/unweighted directed/undirected connection *weight* OR *length* matrix.
        transform (str, optional): If the input matrix is a connection *weight* matrix, specify a transform that maps input connection weights to connection lengths.  Two transforms are available: 'log' -> l_ij = -log(w_ij), 'inv' -> l_ij = 1/w_ij. If the input matrix is a connection *length* matrix, do not specify a transform (or specify an empty transform argument). Defaults to None.


    Returns:
        tuple: A tuple containing:
            SPL (numpy.ndarray): Unweighted/Weighted shortest path-length matrix. If D is a directed matrix, then SPL is not symmetric.
            hops (numpy.ndarray): Number of edges in the shortest path matrix. If D is unweighted, SPL and hops are identical. Only returned if nargout > 1 in MATLAB.
            Pmat (numpy.ndarray): Elements {i,j} of this matrix indicate the next node in the shortest path between i and j. This matrix is used as an input argument for a function that returns the sequence of nodes comprising the shortest path between a given pair of nodes. Only returned if nargout > 1 in MATLAB.

    Notes:
        There may be more than one shortest path between any pair of nodes in the network. Non-unique shortest paths are termed shortest path degeneracies, and are most likely to occur in unweighted networks. When the shortest-path is degenerate, The elements of matrix Pmat correspond to the first shortest path discovered by the algorithm.

        The input matrix may be either a connection weight matrix, or a connection length matrix. The connection length matrix is typically obtained with a mapping from weight to length, such that higher weights are mapped to shorter lengths (see above).

    Algorithm:  Floyd–Warshall Algorithm
    """

    if transform is not None and transform != "":
        if transform == 'log':
            if np.any((D < 0) & (D > 1)):
                raise ValueError('connection-strengths must be in the interval [0,1) to use the transform -log(w_ij)')
            else:
                SPL = -np.log(D)
        elif transform == 'inv':
            SPL = 1.0 / D
        else:
            raise ValueError('Unexpected transform type. Only "log" and "inv" are accepted')
    else:  # the input is a connection lengths matrix.
        SPL = np.copy(D)
        SPL[SPL == 0] = np.inf

    n = D.shape[0]

    flag_find_paths = False
    if nargout > 1:  # Simulate MATLAB's nargout
        flag_find_paths = True
        hops = np.array(D != 0, dtype=float)
        Pmat = np.tile(np.arange(1, n + 1), (n, 1))
    
    for k in range(n):
        i2k_k2j = np.add.outer(SPL[:, k], SPL[k, :])

        if flag_find_paths:
            path = SPL > i2k_k2j
            i, j = np.nonzero(path)
            hops[path] = hops[i, k] + hops[k, j]
            Pmat[path] = Pmat[i, k]

        SPL = np.minimum(SPL, i2k_k2j)

    np.fill_diagonal(SPL, 0)

    if flag_find_paths:
        np.fill_diagonal(hops, 0)
        np.fill_diagonal(Pmat,0)

    if flag_find_paths:
        return SPL, hops, Pmat
    else:
        return SPL


