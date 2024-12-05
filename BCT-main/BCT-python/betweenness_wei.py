# Translated from betweenness_wei.m

import numpy as np

def betweenness_wei(G):
    """
    Node betweenness centrality.

    Calculates the node betweenness centrality, which is the fraction of all shortest paths in 
    the network that contain a given node. Nodes with high values of 
    betweenness centrality participate in a large number of shortest paths.

    Args:
        G: Directed/undirected connection-length matrix.

    Returns:
        BC: Node betweenness centrality vector.

    Notes:
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

    for u in range(n):
        D = np.full(n, np.inf)  # distance from u
        D[u] = 0
        NP = np.zeros(n)  # number of paths from u
        NP[u] = 1
        S = np.full(n, True)  # distance permanence (True is temporary)
        P = np.zeros((n, n), dtype=bool)  # predecessors
        Q = np.zeros(n, dtype=int)
        q = n -1 #order of non-increasing distance

        G1 = np.copy(G)
        V = [u]
        while V:
            v = V.pop(0)
            S[v] = False  # distance u->v is now permanent
            G1[:, v] = 0  # no in-edges as already shortest
            W = np.where(G1[v, :])[0]  # neighbours of v
            for w in W:
                Duw = D[v] + G1[v, w]  # path length to be tested
                if Duw < D[w]:  # if new u->w shorter than old
                    D[w] = Duw
                    NP[w] = NP[v]  # NP(u->w) = NP of new path
                    P[w, :] = False
                    P[w, v] = True  # v is the only predecessor
                elif Duw == D[w]:  # if new u->w equal to old
                    NP[w] += NP[v]  # NP(u->w) sum of old and new
                    P[w, v] = True  # v is also a predecessor

            min_unreached_distance = np.min(D[S])
            if np.isnan(min_unreached_distance):
                break #all nodes reached
            elif np.isinf(min_unreached_distance):
                Q[q:] = np.where(np.isinf(D))[0]
                break # some nodes cannot be reached

            next_nodes = np.where(D == min_unreached_distance)[0]
            V.extend(next_nodes)


        DP = np.zeros(n)  # dependency
        for w in Q[:n-1]:
            BC[w] += DP[w]
            for v in np.where(P[w, :])[0]:
                DP[v] += (1 + DP[w]) * NP[v] / NP[w]

    return BC

