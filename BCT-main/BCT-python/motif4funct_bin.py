# Translated from motif4funct_bin.m

import numpy as np

def motif4funct_bin(A):
    # MOTIF4FUNCT_BIN       Frequency of functional class-4 motifs
    #
    #   [f,F] = motif4funct_bin(A);
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

    M4, ID4, N4 = _load_motif34lib() #load motif data

    n = len(A)                                    #number of vertices in A
    f = np.zeros(199)
    F = np.zeros((199, n))                                 #frequency

    A = (A != 0).astype(int)                                     #adjacency matrix
    As = A | A.T                                       #symmetrized adjacency

    for u in range(n - 3):                                     #loop u 1:n-2
        V1 = np.concatenate((np.zeros(u, dtype=bool), As[u, u+1:]))                #v1: neibs of u (>u)
        for v1 in np.where(V1)[0]:
            V2 = np.concatenate((np.zeros(u, dtype=bool), As[v1, u+1:]))           #v2: all neibs of v1 (>u)
            V2[V1] = False                               #not already in V1
            V2 = V2 | np.concatenate((np.zeros(v1, dtype=bool), As[u, v1+1:]))     #and all neibs of u (>v1)
            for v2 in np.where(V2)[0]:
                vz = max(v1, v2)                      #vz: largest rank node
                V3 = np.concatenate((np.zeros(u, dtype=bool), As[v2, u+1:]))     #v3: all neibs of v2 (>u)
                V3[V2] = False                           #not already in V1&V2
                V3 = V3 | np.concatenate((np.zeros(v2, dtype=bool), As[v1, v2+1:]))#and all neibs of v1 (>v2)
                V3[V1] = False                           #not already in V1
                V3 = V3 | np.concatenate((np.zeros(vz, dtype=bool), As[u, vz+1:])) #and all neibs of u (>vz)
                for v3 in np.where(V3)[0]:

                    a = np.array([A[v1, u], A[v2, u], A[v3, u], A[u, v1], A[v2, v1], A[v3, v1],
                                  A[u, v2], A[v1, v2], A[v3, v2], A[u, v3], A[v1, v3], A[v2, v3]])
                    ind = (M4 @ a) == N4                 #find all contained isomorphs
                    id = ID4[ind]

                    idu, j = np.unique(id, return_inverse=True)             #unique motif occurences
                    j = np.concatenate(([0], j))                        
                    mu = len(idu)                 #number of unique motifs
                    f2 = np.zeros(mu)

                    for h in range(mu):                      #for each unique motif
                        f2[h] = j[h+1] - j[h]              #and frequencies
                    

                    #then add to cumulative count
                    f[idu] += f2
                    if len(f.shape) == 2:
                        F[idu, [u, v1, v2, v3]] += np.tile(f2, (4,1)).T

    return f, F

def _load_motif34lib():
    #This function would typically load data from a file.  Replace this with your actual loading mechanism.
    #For demonstration purposes, we'll return placeholder values.  In a real application, replace this with your actual data loading
    M4 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    ID4 = np.array([1])
    N4 = np.array([1])

    return M4, ID4, N4

