# Translated from density_und.m

import numpy as np

def density_und(CIJ):
    # DENSITY_UND        Density
    #
    #   kden = density_und(CIJ);
    #   [kden,N,K] = density_und(CIJ);
    #
    #   Density is the fraction of present connections to possible connections.
    #
    #   Input:      CIJ,    undirected (weighted/binary) connection matrix
    #
    #   Output:     kden,   density
    #               N,      number of vertices
    #               K,      number of edges
    #
    #   Notes:  Assumes CIJ is undirected and has no self-connections.
    #           Weight information is discarded.
    #
    #
    #   Olaf Sporns, Indiana University, 2002/2007/2008
    #
    # Modification history:
    # 2009-10: K fixed to sum over one half of CIJ [Tony Herdman, SFU]

    N = np.shape(CIJ)[0] # Number of vertices
    K = np.sum(np.triu(CIJ)) # Number of edges (upper triangle only)
    kden = K / ((N**2 - N) / 2) # Density

    return kden, N, K


