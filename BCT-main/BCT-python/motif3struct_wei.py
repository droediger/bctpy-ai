# Translated from motif3struct_wei.m

import numpy as np

def motif3struct_wei(W):
    #MOTIF3STRUCT_WEI       Intensity and coherence of structural class-3 motifs
    #
    #   [I,Q,F] = motif3struct_wei(W);
    #
    #   Structural motifs are patterns of local connectivity in complex
    #   networks. Such patterns are particularly diverse in directed networks.
    #   The motif frequency of occurrence around an individual node is known as
    #   the motif fingerprint of that node. The motif intensity and coherence
    #   are weighted generalizations of the motif frequency. The motif
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
    #       1. The function find_motif34.m outputs the motif legend.
    #       2. Average intensity and coherence are given by I./F and Q./F
    #       3. All weights must be between 0 and 1. This may be achieved using
    #          the weight_conversion.m function, as follows:
    #          W_nrm = weight_conversion(W, 'normalize');
    #
    #   References: Onnela et al. (2005), Phys Rev E 71:065103
    #               Milo et al. (2002) Science 298:824-827
    #               Sporns O, Kötter R (2004) PLoS Biol 2: e369%
    #
    #
    #   Mika Rubinov, UNSW/U Cambridge, 2007-2015
    #
    #   Modification History:
    #   2007: Original
    #   2015: Improved documentation

    M3 = np.array([]) # Initialize persistent variables.  These would be loaded from a file in the original code.
    M3n = np.array([])
    ID3 = np.array([])
    N3 = np.array([])

    if M3.size == 0: # Check if persistent variables are empty
        #load motif34lib M3 M3n ID3 N3         	#load motif data. This line would load data from a file in the original MATLAB code.  This is omitted here as the data is not provided.
        pass # Placeholder for loading motif data

    n = len(W)                                #number of vertices in W
    I = np.zeros((13, n))                              #intensity
    Q = np.zeros((13, n))                              #coherence
    F = np.zeros((13, n))                          	#frequency

    A = 1 * (W != 0)                                 #adjacency matrix
    As = A | A.T                                   #symmetrized adjacency

    for u in range(n - 2):                               	#loop u 1:n-2
        V1 = np.concatenate((np.zeros(u, dtype=bool), As[u, u + 1:]))         	#v1: neibs of u (>u)
        for v1_index in np.nonzero(V1)[0]:
            v1 = v1_index + u +1
            V2 = np.concatenate((np.zeros(u, dtype=bool), As[v1 -1, u + 1:]))       #v2: all neibs of v1 (>u)
            V2[V1] = 0                           #not already in V1
            V2 = np.logical_or(np.concatenate((np.zeros(v1 -1, dtype=bool), As[u, v1:])), V2) #and all neibs of u (>v1)
            for v2_index in np.nonzero(V2)[0]:
                v2 = v2_index + u + 1
                w = np.array([W[v1 - 1, u], W[v2 - 1, u], W[u, v1 - 1], W[v2 - 1, v1 - 1], W[u, v2 - 1], W[v1 - 1, v2 - 1]])
                s = np.sum(10**(5 - np.arange(6)) * np.array([A[v1 - 1, u], A[v2 - 1, u], A[u, v1 - 1], A[v2 - 1, v1 - 1], A[u, v2 - 1], A[v1 - 1, v2 - 1]]).astype(np.uint32))
                ind = (s == M3n) #This will likely fail without the data loaded from motif34lib

                M = w * M3[ind, :]
                id = ID3[ind]
                l = N3[ind]
                x = np.sum(M, axis=1) / l                	#arithmetic mean
                M[M == 0] = 1                      #enable geometric mean
                i = np.prod(M, axis=1)**(1 / l)              #intensity
                q = i / x                          #coherence

                #then add to cumulative count
                I[id, [u, v1 - 1, v2 - 1]] = I[id, [u, v1 - 1, v2 - 1]] + np.array([i, i, i])
                Q[id, [u, v1 - 1, v2 - 1]] = Q[id, [u, v1 - 1, v2 - 1]] + np.array([q, q, q])
                F[id, [u, v1 - 1, v2 - 1]] = F[id, [u, v1 - 1, v2 - 1]] + np.array([1, 1, 1])
    return I, Q, F

