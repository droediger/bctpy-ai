# Translated from kcoreness_centrality_bu.m

import numpy as np

def kcoreness_centrality_bu(CIJ):
    # K-coreness centrality
    #
    #   [coreness,kn] = kcoreness_centrality_bu(CIJ)
    #
    #   The k-core is the largest subgraph comprising nodes of degree at least
    #   k. The coreness of a node is k if the node belongs to the k-core but
    #   not to the (k+1)-core. This function computes the coreness of all nodes
    #   for a given binary undirected connection matrix.
    #
    #   input:          CIJ,        connection/adjacency matrix (binary, undirected)
    #
    #   output:    coreness,        node coreness.
    #                    kn,        size of k-core
    #
    #   References: e.g. Hagmann et al. (2008) PLoS Biology
    #
    #   Olaf Sporns, Indiana University, 2007/2008/2010/2012

    N = CIJ.shape[0]

    # determine if the network is undirected - if not, compute coreness on the
    # corresponding undirected network
    CIJund = CIJ + CIJ.T
    if np.any(CIJund > 1):
        CIJ = (CIJund > 0).astype(int)

    coreness = np.zeros(N)
    kn = np.zeros(N)
    for k in range(1, N + 1):
        CIJkcore, kn[k-1] = kcore_bu(CIJ, k)
        ss = np.sum(CIJkcore, axis=1) > 0
        coreness[ss] = k

    return coreness, kn

def kcore_bu(CIJ,k):
    #placeholder for undefined function kcore_bu
    #This function should be defined elsewhere and it should take a connection matrix and an integer k as input
    # and return a boolean array indicating which nodes are in the k-core and the size of the k-core.
    #This is a dummy implementation for demonstration purposes only.  Replace with your actual implementation.

    N = CIJ.shape[0]
    degrees = np.sum(CIJ, axis=1)
    CIJkcore = np.ones(N, dtype=bool)
    
    while True:
        to_remove = np.where(degrees < k)[0]
        if len(to_remove) == 0:
            break
        CIJkcore[to_remove] = False
        degrees[to_remove] = 0
        degrees = np.sum(CIJ * CIJkcore[:, np.newaxis], axis=1)

    kn = np.sum(CIJkcore)
    return CIJkcore, kn


