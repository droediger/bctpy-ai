# Translated from makelatticeCIJ.m

import numpy as np

def makelatticeCIJ(N, K):
    # MAKELATTICECIJ     Synthetic lattice network
    #
    #   CIJ = makelatticeCIJ(N,K);
    #
    #   This function generates a directed lattice network without toroidal 
    #   boundary conditions (i.e. no ring-like "wrapping around").
    #
    #   Inputs:     N,      number of vertices
    #               K,      number of edges
    #
    #   Outputs:    CIJ,    connection matrix
    #
    #   Note: The lattice is made by placing connections as close as possible 
    #   to the main diagonal, without wrapping around. No connections are made 
    #   on the main diagonal. In/Outdegree is kept approx. constant at K/N.
    #
    #
    #   Olaf Sporns, Indiana University, 2005/2007

    # initialize
    CIJ = np.zeros((N, N))
    CIJ1 = np.ones((N, N))
    KK = 0
    cnt = 0
    seq = np.arange(1, N)

    # fill in
    while (KK < K):
        cnt = cnt + 1
        dCIJ = np.triu(CIJ1, seq[cnt -1]) - np.triu(CIJ1, seq[cnt -1] + 1)
        dCIJ = dCIJ + dCIJ.T
        CIJ = CIJ + dCIJ
        KK = np.sum(CIJ)

    # remove excess connections
    overby = KK - K
    if (overby > 0):
        i, j = np.nonzero(dCIJ)
        rp = np.random.permutation(len(i))
        for ii in range(overby):
            CIJ[i[rp[ii]], j[rp[ii]]] = 0

    return CIJ


