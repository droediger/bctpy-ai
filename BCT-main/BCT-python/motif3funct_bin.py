# Translated from motif3funct_bin.m

import numpy as np

def motif3funct_bin(A):
    # MOTIF3FUNCT_BIN       Frequency of functional class-3 motifs
    #
    #   [f,F] = motif3funct_bin(A);
    #
    #   *Structural motifs* are patterns of local connectivity in complex
    #   networks. In contrast, *functional motifs* are all possible subsets of
    #   patterns of local connectivity embedded within structural motifs. Such
    #   patterns are particularly diverse in directed networks. The motif
    #   frequency of occurrence around an individual node is known as the motif
    #   fingerprint of that node. The total motif frequency of occurrence in
    #   the whole network is correspondingly known as the motif fingerprint of
    #   the network.
    #
    #   Input:      A,      binary directed connection matrix
    #
    #   Output:     F,      node motif frequency fingerprint
    #               f,      network motif frequency fingerprint
    #
    #   Notes: 
    #       1. The function find_motif34.m outputs the motif legend.
    #       2. There is a source of possible confusion in motif terminology.
    #          Motifs ("structural" and "functional") are most frequently
    #          considered only in the context of anatomical brain networks
    #          (Sporns and Kötter, 2004). On the other hand, motifs are not
    #          commonly studied in undirected networks, due to the paucity of
    #          local undirected connectivity patterns.
    #
    #   References: Milo et al. (2002) Science 298:824-827
    #               Sporns O, Kötter R (2004) PLoS Biol 2: e369
    #
    #
    #   Mika Rubinov, UNSW/U Cambridge, 2007-2015

    #   Modification History:
    #   2007: Original
    #   2015: Improved documentation

    M3, ID3, N3 = load_motif34lib() # Load motif data

    n = len(A)                                #number of vertices in A
    f = np.zeros(13, dtype=int)                              #motif count for whole graph
    F = np.zeros((13, n), dtype=int)                          	#frequency

    A = (A != 0).astype(int)                                 #adjacency matrix
    As = A | A.T                                   #symmetrized adjacency

    for u in range(n - 2):                               	#loop u 1:n-2
        V1 = np.concatenate((np.zeros(u, dtype=bool), As[u, u+1:]))         	#v1: neibs of u (>u)
        for v1_index in np.nonzero(V1)[0]:
            v1 = v1_index + u +1
            V2 = np.concatenate((np.zeros(u, dtype=bool), As[v1-1, u+1:]))       #v2: all neibs of v1 (>u)
            V2[V1] = 0                           #not already in V1
            V2 = np.logical_or(np.concatenate((np.zeros(v1-1,dtype=bool), As[u, v1:])),V2) #and all neibs of u (>v1)
            for v2_index in np.nonzero(V2)[0]:
                v2 = v2_index + u + 1
                a = np.array([A[v1-1, u], A[v2-1, u], A[u, v1-1], A[v2-1, v1-1], A[u, v2-1], A[v1-1, v2-1]])
                ind = (M3 @ a) == N3                 #find all contained isomorphs
                id = ID3[ind]

                idu, j = np.unique(id, return_inverse=True)             #unique motif occurences
                j = np.concatenate(([0], j))                        
                mu = len(idu)                 #number of unique motifs
                f2 = np.zeros(mu, dtype=int)

                for h in range(mu):                      #for each unique motif
                    f2[h] = j[h+1] - j[h]              #and frequencies
                

                #then add to cumulative count
                f[idu] += f2
                if len(f.shape) == 2 :
                    F[idu, [u, v1-1, v2-1]] += np.tile(f2,(1,3))

    return f, F


def load_motif34lib():
    #This is a placeholder.  Replace with your actual data loading
    #This example uses example data.  You'll need to load your actual M3, ID3, and N3 data.
    M3 = np.array([[1, 1, 1, 1, 1, 1]])
    ID3 = np.array([1])
    N3 = np.array([6])
    return M3, ID3, N3

