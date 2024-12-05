# Translated from matching_ind_und.m

import numpy as np

def matching_ind_und(CIJ):
    #MATCHING_IND_UND       matching index
    #
    #   M0 = MATCHING_IND_UND(CIJ) computes matching index for undirected
    #   graph specified by adjacency matrix CIJ. Matching index is a measure of
    #   similarity between two nodes' connectivity profiles (excluding their
    #   mutual connection, should it exist).
    #
    #   Inputs:     CIJ,    undirected adjacency matrix
    #
    #   Outputs:    M0,     matching index matrix.
    #
    #   Richard Betzel, Indiana University, 2013
    #

    CIJ0 = CIJ.copy()
    K = np.sum(CIJ0, axis=1)
    R = K != 0
    N = np.sum(R)
    CIJ = CIJ0[R,:][:,R]
    I = np.ones((N,N), dtype=bool) - np.eye(N, dtype=bool)
    M = np.zeros((N,N))
    for i in range(N):
        c1 = CIJ[i,:]
        use = np.logical_or(c1[:, np.newaxis], CIJ)
        use[:,i] = False
        use = use*I

        ncon1 = use*c1[:,np.newaxis]
        ncon2 = use*CIJ
        ncon = np.sum(ncon1 + ncon2, axis=0)

        M[:,i] = 2*np.sum(np.logical_and(ncon1, ncon2), axis=0)/ncon

    M = M*I
    M[np.isnan(M)] = 0
    M0 = np.zeros_like(CIJ0)
    M0[R,:][:,R] = M

    return M0


