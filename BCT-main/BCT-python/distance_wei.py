# Translated from distance_wei.m

import numpy as np

def distance_wei(L):
    # DISTANCE_WEI       Distance matrix (Dijkstra's algorithm)
    #
    #   D = distance_wei(L);
    #   [D,B] = distance_wei(L);
    #
    #   The distance matrix contains lengths of shortest paths between all
    #   pairs of nodes. An entry (u,v) represents the length of shortest path 
    #   from node u to node v. The average shortest path length is the 
    #   characteristic path length of the network.
    #
    #   Input:      L,      Directed/undirected connection-length matrix.
    #   *** NB: The length matrix L isn't the weights matrix W (see below) ***
    #
    #   Output:     D,      distance (shortest weighted path) matrix
    #               B,      number of edges in shortest weighted path matrix
    #
    #   Notes:
    #       The input matrix must be a connection-length matrix, typically
    #   obtained via a mapping from weight to length. For instance, in a
    #   weighted correlation network higher correlations are more naturally
    #   interpreted as shorter distances and the input matrix should
    #   consequently be some inverse of the connectivity matrix. 
    #       The number of edges in shortest weighted paths may in general 
    #   exceed the number of edges in shortest binary paths (i.e. shortest
    #   paths computed on the binarized connectivity matrix), because shortest 
    #   weighted paths have the minimal weighted distance, but not necessarily 
    #   the minimal number of edges.
    #       Lengths between disconnected nodes are set to Inf.
    #       Lengths on the main diagonal are set to 0.
    #
    #   Algorithm: Dijkstra's algorithm.
    #
    #
    #   Mika Rubinov, UNSW/U Cambridge, 2007-2012.
    #   Rick Betzel and Andrea Avena, IU, 2012

    n = len(L)
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0)  #distance matrix
    B = np.zeros((n, n))  #number of edges matrix

    for u in range(n):
        S = np.ones(n, dtype=bool)  #distance permanence (True is temporary)
        L1 = np.copy(L)
        V = u
        while True:
            S[V] = False  #distance u->V is now permanent
            L1[:, V] = 0  #no in-edges as already shortest
            for v in [V]: #Note: the original code iterates only once here, this is corrected
                T = np.where(L1[v, :] > 0)[0]  #neighbours of shortest nodes
                if len(T) > 0:
                    d = np.minimum(D[u, T], D[u, v] + L1[v, T])
                    D[u, T] = d
                    ind = T[np.where(d == D[u,v] + L1[v,T])[0]] #indices of lengthened paths
                    B[u, ind] = B[u, v] + 1  #increment no. of edges in lengthened paths

            minD = np.min(D[u, S])
            if np.isnan(minD) or np.isinf(minD): #isempty: all nodes reached;
                break  #isinf: some nodes cannot be reached

            V = np.where(D[u, :] == minD)[0][0]
    return D, B

