# Translated from edge_betweenness_bin.m

import numpy as np

def edge_betweenness_bin(G):
    #EDGE_BETWEENNESS_BIN    Edge betweenness centrality
    #
    #   EBC = edge_betweenness_bin(A);
    #   [EBC BC] = edge_betweenness_bin(A);
    #
    #   Edge betweenness centrality is the fraction of all shortest paths in 
    #   the network that contain a given edge. Edges with high values of 
    #   betweenness centrality participate in a large number of shortest paths.
    #
    #   Input:      A,      binary (directed/undirected) connection matrix.
    #
    #   Output:     EBC,    edge betweenness centrality matrix.
    #               BC,     node betweenness centrality vector.
    #
    #   Note: Betweenness centrality may be normalised to the range [0,1] as
    #   BC/[(N-1)(N-2)], where N is the number of nodes in the network.
    #
    #   Reference: Brandes (2001) J Math Sociol 25:163-177.
    #
    #
    #   Mika Rubinov, UNSW/U Cambridge, 2007-2012

    n = len(G)
    BC = np.zeros((n, 1))                  #vertex betweenness
    EBC = np.zeros((n, n))                   #edge betweenness

    for u in range(n):
        D = np.zeros((1, n), dtype=bool)
        D[0, u] = 1      	#distance from u
        NP = np.zeros((1, n))
        NP[0, u] = 1     #number of paths from u
        P = np.zeros((n, n), dtype=bool)                 #predecessors
        Q = np.zeros((1, n))
        q = n          #order of non-increasing distance

        Gu = np.copy(G)
        V = [u]
        while V:
            Gu[:, V] = 0              #remove remaining in-edges
            new_V = []
            for v in V:
                Q[0, q - 1] = v
                q -= 1
                W = np.where(Gu[v, :] == 1)[0]                #neighbours of v
                for w in W:
                    if D[0, w]:
                        NP[0, w] = NP[0, w] + NP[0, v]      #NP(u->w) sum of old and new
                        P[w, v] = 1               #v is a predecessor
                    else:
                        D[0, w] = 1
                        NP[0, w] = NP[0, v]            #NP(u->w) = NP of new path
                        P[w, v] = 1               #v is a predecessor
                    new_V.append(w)

            V = list(set(new_V))
            

        if not np.all(D):                              #if some vertices unreachable,
            unreached = np.where(D[0,:] == 0)[0]                    #...these are first-in-line
            Q[0, :q] = unreached
            

        DP = np.zeros((n, 1))                          #dependency
        for w_index in range(n - 1):
            w = int(Q[0, w_index])
            BC[w] = BC[w] + DP[w]
            v_indices = np.where(P[w, :] == 1)[0]
            for v_index in v_indices:
                v = v_index
                DPvw = (1 + DP[w]) * NP[0, v] / NP[0, w]
                DP[v] = DP[v] + DPvw
                EBC[v, w] = EBC[v, w] + DPvw

    return EBC, BC.flatten()

