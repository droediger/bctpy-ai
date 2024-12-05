# Translated from density_dir.m

import numpy as np

def density_dir(CIJ):
    # DENSITY_DIR        Density
    #
    #   kden = density_dir(CIJ);
    #   [kden,N,K] = density_dir(CIJ);
    #
    #   Density is the fraction of present connections to possible connections.
    #
    #   Input:      CIJ,    directed (weighted/binary) connection matrix
    #
    #   Output:     kden,   density
    #               N,      number of vertices
    #               K,      number of edges
    #
    #   Notes:  Assumes CIJ is directed and has no self-connections.
    #           Weight information is discarded.
    #
    #
    #   Olaf Sporns, Indiana University, 2002/2007/2008

    N = np.shape(CIJ)[0] # Number of vertices
    K = np.count_nonzero(CIJ) # Number of edges
    kden = K/(N**2-N) # Density

    return kden, N, K


