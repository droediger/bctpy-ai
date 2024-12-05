# Translated from distance_bin.m

import numpy as np

def distance_bin(A):
    #DISTANCE_BIN       Distance matrix
    #
    #   D = distance_bin(A);
    #
    #   The distance matrix contains lengths of shortest paths between all
    #   pairs of nodes. An entry (u,v) represents the length of shortest path 
    #   from node u to node v. The average shortest path length is the 
    #   characteristic path length of the network.
    #
    #   Input:      A,      binary directed/undirected connection matrix
    #
    #   Output:     D,      distance matrix
    #
    #   Notes: 
    #       Lengths between disconnected nodes are set to Inf.
    #       Lengths on the main diagonal are set to 0.
    #
    #   Algorithm: Algebraic shortest paths.
    #
    #
    #   Mika Rubinov, U Cambridge
    #   Jonathan Clayden, UCL
    #   2007-2013

    # Modification history:
    # 2007: Original (MR)
    # 2013: Bug fix, enforce zero distance for self-connections (JC)

    A = (A != 0).astype(float)          #binarize and convert to double format

    l = 1                             #path length
    Lpath = A                         #matrix of paths l
    D = A                             #distance matrix

    Idx = np.ones(A.shape, dtype=bool)
    while np.any(Idx):
        l += 1
        Lpath = np.dot(Lpath, A)
        Idx = (Lpath != 0) & (D == 0)
        D[Idx] = l
    
    D[D == 0] = np.inf                 #assign inf to disconnected nodes
    np.fill_diagonal(D, 0)             #clear diagonal

    return D


