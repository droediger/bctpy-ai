# Translated from efficiency_wei.m

import numpy as np

def efficiency_wei(W, local=0):
    """Global efficiency, local efficiency.

    Args:
        W (numpy.ndarray): weighted undirected or directed connection matrix
        local (int, optional): optional argument. 
                              local=0 computes the global efficiency (default).
                              local=1 computes the original version of the local efficiency.
                              local=2 computes the modified version of the local efficiency (recommended).

    Returns:
        numpy.ndarray: global efficiency (scalar) or local efficiency (vector)

    Notes:
        The efficiency is computed using an auxiliary connection-length matrix L, defined as L_ij = 1/W_ij for all nonzero L_ij. This has an intuitive interpretation, as higher connection weights intuitively correspond to shorter lengths.
        The weighted local efficiency broadly parallels the weighted clustering coefficient of Onnela et al. (2005) and distinguishes the influence of different paths based on connection weights of the corresponding neighbors to the node in question. In other words, a path between two neighbors with strong connections to the node in question contributes more to the local efficiency than a path between two weakly connected neighbors. Note that the original weighted variant of the local efficiency (described in Rubinov and Sporns, 2010) is not a true generalization of the binary variant, while the modified variant (described in Wang et al., 2016) is a true generalization.
        For ease of interpretation of the local efficiency it may be advantageous to rescale all weights to lie between 0 and 1.

    Algorithm:  Dijkstra's algorithm

    References: Latora and Marchiori (2001) Phys Rev Lett 87:198701.
                Onnela et al. (2005) Phys Rev E 71:065103
                Fagiolo (2007) Phys Rev E 76:026107.
                Rubinov M, Sporns O (2010) NeuroImage 52:1059-69
                Wang Y et al. (2016) Neural Comput 21:1-19.
    """
    n = len(W)  # number of nodes
    ot = 1 / 3  # one third

    L = np.copy(W)  # connection-length matrix
    A = W > 0  # adjacency matrix
    L[A] = 1 / L[A]
    A = A.astype(float)

    if local:  # local efficiency
        E = np.zeros(n)
        cbrt_W = W**ot
        if local == 1:
            for u in range(n):
                V = np.where((A[u, :] | A[:, u].T).flatten())[0]  # neighbors
                sw = cbrt_W[u, V] + cbrt_W[V, u].T  # symmetrized weights vector
                di = distance_inv_wei(L[np.ix_(V, V)])  # inverse distance matrix
                se = di**ot + di.T**ot  # symmetrized inverse distance matrix
                numer = np.sum(((sw.T * sw) * se)) / 2  # numerator
                if numer != 0:
                    sa = A[u, V] + A[V, u].T  # symmetrized adjacency vector
                    denom = np.sum(sa)**2 - np.sum(sa**2)  # denominator
                    E[u] = numer / denom  # local efficiency
        elif local == 2:
            cbrt_L = L**ot
            for u in range(n):
                V = np.where((A[u, :] | A[:, u].T).flatten())[0]  # neighbors
                sw = cbrt_W[u, V] + cbrt_W[V, u].T  # symmetrized weights vector
                di = distance_inv_wei(cbrt_L[np.ix_(V, V)])  # inverse distance matrix
                se = di + di.T  # symmetrized inverse distance matrix
                numer = np.sum(((sw.T * sw) * se)) / 2  # numerator
                if numer != 0:
                    sa = A[u, V] + A[V, u].T  # symmetrized adjacency vector
                    denom = np.sum(sa)**2 - np.sum(sa**2)  # denominator
                    E[u] = numer / denom  # local efficiency
    else:
        di = distance_inv_wei(L)
        E = np.sum(di) / (n**2 - n)  # global efficiency
    return E


def distance_inv_wei(W_):
    n_ = len(W_)
    D = np.full((n_, n_), np.inf)  # distance matrix
    np.fill_diagonal(D, 0)

    for u in range(n_):
        S = np.ones(n_, dtype=bool)  # distance permanence (true is temporary)
        W1_ = np.copy(W_)
        V = [u]
        while True:
            S[V] = False  # distance u->V is now permanent
            W1_[:, V] = 0  # no in-edges as already shortest
            for v in V:
                T = np.where(W1_[v, :])[0]  # neighbors of shortest nodes
                D[u, T] = np.minimum(D[u, T], D[u, v] + W1_[v, T])  # smallest of old/new path lengths

            minD = np.min(D[u, S])
            if minD is np.inf or not np.any(S):  # isempty: all nodes reached; isinf: some nodes cannot be reached
                break

            V = np.where(D[u, :] == minD)[0]
    D = 1 / D  # invert distance
    np.fill_diagonal(D, 0)
    return D

