# Translated from makeringlatticeCIJ.m

import numpy as np

def makeringlatticeCIJ(N, K):
    # makeringlatticeCIJ     Synthetic lattice network
    #
    #   CIJ = makeringlatticeCIJ(N,K);
    #
    #   This function generates a directed lattice network with toroidal 
    #   boundary conditions (i.e. with ring-like "wrapping around").
    #
    #   Inputs:     N,      number of vertices
    #               K,      number of edges
    #
    #   Outputs:    CIJ,    connection matrix
    #
    #   Note: The lattice is made by placing connections as close as possible 
    #   to the main diagonal, with wrapping around. No connections are made 
    #   on the main diagonal. In/Outdegree is kept approx. constant at K/N.
    #
    #
    #   Olaf Sporns, Indiana University, 2005/2007

    # Initialize
    CIJ = np.zeros((N, N))
    CIJ1 = np.ones((N, N))
    KK = 0
    cnt = 0
    seq = np.arange(1, N)
    seq2 = np.arange(N - 1, 0, -1)

    # Fill in
    while KK < K:
        cnt += 1
        dCIJ = np.triu(CIJ1, seq[cnt - 1]) - np.triu(CIJ1, seq[cnt - 1] + 1)
        dCIJ2 = np.triu(CIJ1, seq2[cnt - 1]) - np.triu(CIJ1, seq2[cnt - 1] + 1)
        dCIJ = dCIJ + dCIJ.T + dCIJ2 + dCIJ2.T
        CIJ = CIJ + dCIJ
        KK = np.sum(CIJ)

    # Remove excess connections
    overby = KK - K
    if overby > 0:
        i, j = np.nonzero(dCIJ)
        rp = np.random.permutation(len(i))
        for ii in range(overby):
            CIJ[i[rp[ii]], j[rp[ii]]] = 0

    return CIJ


