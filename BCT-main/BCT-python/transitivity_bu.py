# Translated from transitivity_bu.m

import numpy as np

def transitivity_bu(A):
    # TRANSITIVITY_BU    Transitivity
    #
    #   T = transitivity_bu(A);
    #
    #   Transitivity is the ratio of 'triangles to triplets' in the network.
    #   (A classical version of the clustering coefficient).
    #
    #   Input:      A       binary undirected connection matrix (NumPy array)
    #
    #   Output:     T       transitivity scalar
    #
    #   Reference: e.g. Humphries et al. (2008) Plos ONE 3: e0002051
    #
    #
    #   Adapted from Alexandros Goulas, Maastricht University, 2010

    C_tri = np.trace(np.linalg.matrix_power(A, 3)) / (np.sum(np.linalg.matrix_power(A, 2)) - np.trace(np.linalg.matrix_power(A, 2)))

    return C_tri


