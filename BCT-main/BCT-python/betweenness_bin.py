# Translated from betweenness_bin.m

import numpy as np

def betweenness_bin(G):
    """
    Node betweenness centrality.

    Node betweenness centrality is the fraction of all shortest paths in 
    the network that contain a given node. Nodes with high values of 
    betweenness centrality participate in a large number of shortest paths.

    Parameters
    ----------
    G : numpy.ndarray
        Binary (directed/undirected) connection matrix.

    Returns
    -------
    BC : numpy.ndarray
        Node betweenness centrality vector.

    Notes
    -----
    Betweenness centrality may be normalised to the range [0,1] as
    BC/[(N-1)(N-2)], where N is the number of nodes in the network.

    Reference: Kintali (2008) arXiv:0809.1906v2 [cs.DS]
               (generalization to directed and disconnected graphs)
    """
    n = len(G)  # number of nodes
    I = np.eye(n, dtype=bool)  # logical identity matrix
    d = 1  # path length
    NPd = G  # number of paths of length |d|
    NSPd = NPd  # number of shortest paths of length |d|
    NSP = NSPd.copy()
    NSP[I] = 1  # number of shortest paths of any length
    L = NSPd.copy()
    L[I] = 1  # length of shortest paths

    # calculate NSP and L
    while np.any(NSPd):
        d += 1
        NPd = np.dot(NPd, G)
        NSPd = NPd * (L == 0)
        NSP += NSPd
        L += d * (NSPd != 0)

    L[~L.astype(bool)] = np.inf
    L[I] = 0  # L for disconnected vertices is inf
    NSP[~NSP.astype(bool)] = 1  # NSP for disconnected vertices is 1

    Gt = G.T
    DP = np.zeros((n, n))  # vertex on vertex dependency
    diam = d - 1  # graph diameter

    # calculate DP
    for d in range(diam, 1, -1):
        DPd1 = (((L == d) * (1 + DP) / NSP) * Gt) * ((L == (d - 1)) * NSP)
        DP += DPd1  # DPd1: dependencies on vertices |d-1| from source

    BC = np.sum(DP, axis=0)  # compute betweenness

    return BC

