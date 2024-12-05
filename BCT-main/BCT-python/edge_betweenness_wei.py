# Translated from edge_betweenness_wei.m

import numpy as np

def edge_betweenness_wei(G):
    """
    Edge betweenness centrality.

    Parameters
    ----------
    G : numpy.ndarray
        Directed/undirected connection-length matrix.

    Returns
    -------
    EBC : numpy.ndarray
        Edge betweenness centrality matrix.
    BC : numpy.ndarray
        Nodal betweenness centrality vector.

    Notes
    -----
    The input matrix must be a connection-length matrix, typically
    obtained via a mapping from weight to length. For instance, in a
    weighted correlation network higher correlations are more naturally
    interpreted as shorter distances and the input matrix should
    consequently be some inverse of the connectivity matrix.
    Betweenness centrality may be normalised to the range [0,1] as
    BC/[(N-1)(N-2)], where N is the number of nodes in the network.

    Reference: Brandes (2001) J Math Sociol 25:163-177.
    """

    n = len(G)
    BC = np.zeros(n)  # vertex betweenness
    EBC = np.zeros((n, n))  # edge betweenness

    for u in range(n):
        D = np.full(n, np.inf)  # distance from u
        D[u] = 0
        NP = np.zeros(n)  # number of paths from u
        NP[u] = 1
        S = np.ones(n, dtype=bool)  # distance permanence (True is temporary)
        P = np.zeros((n, n), dtype=bool)  # predecessors
        Q = np.zeros(n, dtype=int)
        q = n -1 # order of non-increasing distance

        G1 = np.copy(G)
        V = [u]
        while True:
            S[V] = False  # distance u->V is now permanent
            G1[:,V] = 0  # no in-edges as already shortest

            for v in V:
                Q[q] = v
                q -= 1
                W = np.where(G1[v,:] > 0)[0]  # neighbours of v
                for w in W:
                    Duw = D[v] + G1[v, w]  # path length to be tested
                    if Duw < D[w]:  # if new u->w shorter than old
                        D[w] = Duw
                        NP[w] = NP[v]  # NP(u->w) = NP of new path
                        P[w,:] = False
                        P[w,v] = True  # v is the only predecessor
                    elif Duw == D[w]:  # if new u->w equal to old
                        NP[w] = NP[w] + NP[v]  # NP(u->w) sum of old and new
                        P[w,v] = True  # v is also a predecessor

            minD = np.min(D[S])
            if np.isnan(minD) or minD == np.inf:
                break  # all nodes reached, or ...some cannot be reached
            V = np.where(D == minD)[0]
           

        DP = np.zeros(n)  # dependency
        for w in Q[:n-1]:
            BC[w] = BC[w] + DP[w]
            for v in np.where(P[w,:])[0]:
                DPvw = (1 + DP[w]) * NP[v] / NP[w]
                DP[v] = DP[v] + DPvw
                EBC[v, w] = EBC[v, w] + DPvw

    return EBC, BC


