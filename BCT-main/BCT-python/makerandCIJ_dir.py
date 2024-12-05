# Translated from makerandCIJ_dir.m

import numpy as np

def makerandCIJ_dir(N, K):
    # MAKERANDCIJ_DIR        Synthetic directed random network
    #
    #   CIJ = makerandCIJ_dir(N,K);
    #
    #   This function generates a directed random network
    #
    #   Inputs:     N,      number of vertices
    #               K,      number of edges
    #
    #   Output:     CIJ,    directed random connection matrix
    #
    #   Note: no connections are placed on the main diagonal.
    #
    #
    # Olaf Sporns, Indiana University, 2007/2008

    ind = np.ones((N, N), dtype=bool) - np.eye(N, dtype=bool) # Create a boolean array with ones everywhere except the main diagonal
    i = np.where(ind) #Get the indices where ind is true
    i = np.ravel_multi_index(i, (N,N)) #Convert to linear indices

    rp = np.random.permutation(len(i)) # Randomly permute the indices
    irp = i[rp] # Apply the permutation

    CIJ = np.zeros((N, N)) #Initialize the connectivity matrix
    CIJ[np.unravel_index(irp[:K], (N,N))] = 1 #Set K randomly selected edges to 1

    return CIJ



