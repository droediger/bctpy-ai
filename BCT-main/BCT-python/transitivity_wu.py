# Translated from transitivity_wu.m

import numpy as np

def transitivity_wu(W):
    # TRANSITIVITY_WU    Transitivity
    #
    #   T = transitivity_wu(W);
    #
    #   Transitivity is the ratio of 'triangles to triplets' in the network.
    #   (A classical version of the clustering coefficient).
    #
    #   Input:      W       weighted undirected connection matrix
    #
    #   Output:     T       transitivity scalar
    #
    #   Note:      All weights must be between 0 and 1.
    #              This may be achieved using the weight_conversion function,
    #              W_nrm = weight_conversion(W, 'normalize');
    #
    #   Reference: Rubinov M, Sporns O (2010) NeuroImage 52:1059-69
    #              based on Onnela et al. (2005) Phys Rev E 71:065103
    #
    #
    #   Mika Rubinov, UNSW/U Cambridge, 2010-2015

    #   Modification history:
    #   2010: Original
    #   2015: Expanded documentation

    K = np.sum(W != 0, axis=1)              # Degree (number of non-zero connections) for each node
    cyc3 = np.diag(np.power(np.power(W, 1/3),3)) # Number of triangles for each node.  Uses element-wise power.
    T = np.sum(cyc3) / np.sum(K * (K - 1))     # Transitivity: sum of triangles / sum of possible triplets

    return T


