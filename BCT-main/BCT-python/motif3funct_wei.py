# Translated from motif3funct_wei.m

import numpy as np

def motif3funct_wei(W):
    # MOTIF3FUNCT_WEI       Intensity and coherence of functional class-3 motifs
    #
    #   [I,Q,F] = motif3funct_wei(W);
    #
    #   *Structural motifs* are patterns of local connectivity in complex
    #   networks. In contrast, *functional motifs* are all possible subsets of
    #   patterns of local connectivity embedded within structural motifs. Such
    #   patterns are particularly diverse in directed networks. The motif
    #   frequency of occurrence around an individual node is known as the motif
    #   fingerprint of that node. The motif intensity and coherence are
    #   weighted generalizations of the motif frequency. The motif
    #   intensity is equivalent to the geometric mean of weights of links
    #   comprising each motif. The motif coherence is equivalent to the ratio
    #   of geometric and arithmetic means of weights of links comprising each
    #   motif.  
    #
    #   Input:      W,      weighted directed connection matrix
    #                       (all weights must be between 0 and 1)
    #
    #   Output:     I,      node motif intensity fingerprint
    #               Q,      node motif coherence fingerprint
    #               F,      node motif frequency fingerprint
    #
    #   Notes: 
    #       1. The function find_motif34.m outputs the motif legend.  This is assumed to be pre-loaded.
    #       2. Average intensity and coherence are given by I./F and Q./F
    #       3. All weights must be between 0 and 1. This may be achieved using
    #          the weight_conversion.m function, as follows: 
    #          W_nrm = weight_conversion(W, 'normalize');
    #   	4. There is a source of possible confusion in motif terminology.
    #          Motifs ("structural" and "functional") are most frequently
    #          considered only in the context of anatomical brain networks
    #          (Sporns and Kötter, 2004). On the other hand, motifs are not
    #          commonly studied in undirected networks, due to the paucity of
    #          local undirected connectivity patterns.
    #
    #   References: Onnela et al. (2005), Phys Rev E 71:065103
    #               Milo et al. (2002) Science 298:824-827
    #               Sporns O, Kötter R (2004) PLoS Biol 2: e369
    #
    #
    #   Mika Rubinov, UNSW/U Cambridge, 2007-2015

    #   Modification History:
    #   2007: Original
    #   2015: Improved documentation

    M3 = np.load('motif34lib.npz')['M3'] #load motif data. Assumes data is saved in a .npz file.
    ID3 = np.load('motif34lib.npz')['ID3']
    N3 = np.load('motif34lib.npz')['N3']

    n = len(W)                                 #number of vertices in W
    I = np.zeros((13, n))                     #intensity
    Q = np.zeros((13, n))                     #coherence
    F = np.zeros((13, n))                     #frequency

    A = 1 * (W != 0)                           #adjacency matrix
    As = A | A.T                              #symmetrized adjacency

    for u in range(n - 2):                    #loop u 1:n-2
        V1 = np.concatenate((np.zeros(u, dtype=bool), As[u, u + 1:]))  #v1: neibs of u (>u)
        for v1_index in np.nonzero(V1)[0]:
            v1 = v1_index + u +1
            V2 = np.concatenate((np.zeros(u, dtype=bool), As[v1 -1, u + 1:]))       #v2: all neibs of v1 (>u)
            V2[V1] = 0                           #not already in V1
            V2 = (np.concatenate((np.zeros(v1 -1, dtype=bool), As[u, v1:]))) | V2  #and all neibs of u (>v1)
            for v2_index in np.nonzero(V2)[0]:
                v2 = v2_index + u + 1
                w = np.array([W[v1 - 1, u], W[v2 - 1, u], W[u, v1 - 1], W[v2 - 1, v1 - 1], W[u, v2 - 1], W[v1 - 1, v2 - 1]])
                a = np.array([A[v1 - 1, u], A[v2 - 1, u], A[u, v1 - 1], A[v2 - 1, v1 - 1], A[u, v2 - 1], A[v1 - 1, v2 - 1]])
                ind = (M3 @ a) == N3                 #find all contained isomorphs
                m = np.sum(ind)                     #number of isomorphs

                M = M3[ind, :] * np.tile(w, (m, 1))
                id = ID3[ind]
                l = N3[ind]

                x = np.sum(M, axis=1) / l                  #arithmetic mean
                M[M == 0] = 1                      #enable geometric mean
                i = np.prod(M, axis=1) ** (1 / l)            #intensity
                q = i / x                         #coherence

                idu, j = np.unique(id, return_inverse=True)             #unique motif occurences
                j = np.concatenate(([0], j))                        
                mu = len(idu)                 #number of unique motifs
                i2 = np.zeros(mu)
                q2 = np.zeros(mu)
                f2 = np.zeros(mu)

                for h in range(mu):                      #for each unique motif
                    i2[h] = np.sum(i[j[h]:j[h+1]])    #sum all intensities,
                    q2[h] = np.sum(q[j[h]:j[h+1]])    #coherences
                    f2[h] = j[h+1] - j[h]              #and frequencies
                
                I[idu, [u, v1 - 1, v2 - 1]] += np.array([i2, i2, i2])
                Q[idu, [u, v1 - 1, v2 - 1]] += np.array([q2, q2, q2])
                F[idu, [u, v1 - 1, v2 - 1]] += np.array([f2, f2, f2])

    return I, Q, F

