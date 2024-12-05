# Translated from agreement_weighted.m

import numpy as np

def agreement_weighted(CI, Wts):
    # AGREEMENT_WEIGHTED     Weights agreement matrix
    #
    #   D = AGREEMENT_WEIGHTED(CI,WTS) is identical to AGREEMENT, with the 
    #   exception that each partition's contribution is weighted according to 
    #   the corresponding scalar value stored in the vector WTS. As an example,
    #   suppose CI contained partitions obtained using some heuristic for 
    #   maximizing modularity. A possible choice for WTS might be the Q metric
    #   (Newman's modularity score). Such a choice would add more weight to 
    #   higher modularity partitions.
    #
    #   NOTE: Unlike AGREEMENT, this script does not have the input argument
    #   BUFFSZ.
    #
    #   Inputs:     CI,     set of partitions
    #               WTS,    relative weight of importance of each partition
    #
    #   Outputs:    D,      weighted agreement matrix
    #
    #   Richard Betzel, Indiana University, 2013

    Wts = Wts / np.sum(Wts)
    N, M = CI.shape
    D = np.zeros((N, N))
    for i in range(M):
        d = dummyvar(CI[:, i]) #Assumes dummyvar is defined elsewhere
        D = D + (d @ d.T) * Wts[i]
    return D

def dummyvar(x):
    #This is a placeholder for the dummyvar function.  Replace with actual implementation if available.
    return np.eye(len(np.unique(x)))[x - 1] # Assumes x contains only positive integers. Adjust if needed.



