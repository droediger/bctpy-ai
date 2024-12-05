# Translated from makerandCIJ_und.m

import numpy as np

def makerandCIJ_und(N, K):
    # MAKERANDCIJ_UND        Synthetic directed random network
    #
    #   CIJ = makerandCIJ_und(N,K);
    #
    #   This function generates an undirected random network
    #
    #   Inputs:     N,      number of vertices
    #               K,      number of edges
    #
    #   Output:     CIJ,    undirected random connection matrix
    #
    #   Note: no connections are placed on the main diagonal.
    #
    #
    # Olaf Sporns, Indiana University, 2007/2008

    ind = np.triu(~np.eye(N))
    i = np.where(ind)
    i = np.ravel_multi_index(i,ind.shape)
    rp = np.random.permutation(len(i))
    irp = i[rp]

    CIJ = np.zeros((N,N))
    CIJ[np.unravel_index(irp[:K],CIJ.shape)] = 1
    CIJ = CIJ + CIJ.T         # symmetrize

    return CIJ


