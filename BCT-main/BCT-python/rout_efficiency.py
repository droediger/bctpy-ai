# Translated from rout_efficiency.m

import numpy as np

def rout_efficiency(D, transform=None):
    """
    Mean, pair-wise and local routing efficiency

    Parameters
    ----------
    D : numpy.ndarray
        Weighted/unweighted directed/undirected connection *weight* OR *length* matrix.
    transform : str, optional
        If the input matrix is a connection *weight* matrix, specify a transform that maps input connection weights to connection lengths. 
        Two transforms are available: 'log' -> l_ij = -log(w_ij), 'inv' -> l_ij = 1/w_ij.
        If the input matrix is a connection *length* matrix, do not specify a transform (or specify an empty transform argument).  The default is None.

    Returns
    -------
    GErout : float
        Mean global routing efficiency (scalar).
    Erout : numpy.ndarray
        Pair-wise routing efficiency (matrix).
    Eloc : numpy.ndarray, optional
        Local efficiency (vector). Only returned if requested.

    Notes
    -----
    The input matrix may be either a connection weight matrix, or a connection length matrix. The connection length matrix is typically obtained with a mapping from weight to length, such that higher weights are mapped to shorter lengths (see above).
    Algorithm: Floyd–Warshall Algorithm
    References:
        Latora and Marchiori (2001) Phys Rev Lett
        Goñi et al (2013) PLoS ONE
        Avena-Koenigsberger et al (2016) Brain Structure and Function
    """

    #Assume distance_wei_floyd is defined elsewhere
    n = len(D)  # number of nodes

    Erout = distance_wei_floyd(D, transform)  # pair-wise routing efficiency
    Erout = 1 / Erout
    np.fill_diagonal(Erout, 0)
    GErout = np.sum(Erout[~np.eye(n,dtype=bool)]) / (n**2 - n)  # global routing efficiency

    if len(np.atleast_1d(transform)) == 3 : #Check if a third output is needed.
        Eloc = np.zeros(n)
        for u in range(n):
            Gu = np.where(D[u, :] != 0)[0] # u's neighbors, only considers non-zero connections in this case
            nGu = len(Gu)
            if nGu > 1: #Avoid error if the node has less than 2 neighbours
                e = distance_wei_floyd(D[np.ix_(Gu, Gu)], transform)
                Eloc[u] = np.sum(1 / e[~np.eye(nGu, dtype=bool)]) / nGu  # efficiency of subgraph Gu
    return GErout, Erout, Eloc

def distance_wei_floyd(D,transform): #Assume this function is defined elsewhere.  This would contain a Floyd-Warshall implementation.
    """
    Placeholder for the distance_wei_floyd function.  Replace with your actual implementation.
    This would ideally take the distance matrix and the transform and return the shortest path lengths between all pairs of nodes.
    """
    #Replace this with your actual Floyd-Warshall implementation
    if transform == 'log':
        D = -np.log(D)
    elif transform == 'inv':
        D = 1/D
    n = D.shape[0]
    dist = np.copy(D)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
    return dist


