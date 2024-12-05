# Translated from findwalks.m

import numpy as np

def findwalks(CIJ):
    #FINDWALKS      Network walks
    #
    #   [Wq,twalk,wlq] = findwalks(CIJ);
    #
    #   Walks are sequences of linked nodes, that may visit a single node more
    #   than once. This function finds the number of walks of a given length, 
    #   between any two nodes.
    #
    #   Input:      CIJ         binary (directed/undirected) connection matrix
    #
    #   Outputs:    Wq          3D matrix, Wq(i,j,q) is the number of walks
    #                           from 'i' to 'j' of length 'q'.
    #               twalk       total number of walks found
    #               wlq         walk length distribution as function of 'q'
    #
    #   Notes: Wq grows very quickly for larger N,K,q. Weights are discarded.
    #
    #   Algorithm: algebraic path count
    #
    #
    #   Olaf Sporns, Indiana University, 2002/2007/2008

    # ensure CIJ is binary...
    CIJ = (CIJ != 0).astype(float)

    N = CIJ.shape[0]
    Wq = np.zeros((N, N, N))
    CIJpwr = CIJ.copy()
    Wq[:, :, 0] = CIJ
    for q in range(1, N):
        CIJpwr = CIJpwr @ CIJ
        Wq[:, :, q] = CIJpwr

    # total number of walks
    twalk = np.sum(Wq)

    # walk length distribution
    wlq = np.sum(np.sum(Wq, axis=0), axis=0)

    return Wq, twalk, wlq


