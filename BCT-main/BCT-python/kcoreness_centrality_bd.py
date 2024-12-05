# Translated from kcoreness_centrality_bd.m

import numpy as np

def kcoreness_centrality_bd(CIJ):
    # K-coreness centrality
    #
    #   [coreness,kn] = kcoreness_centrality_bd(CIJ)
    #
    #   The k-core is the largest subgraph comprising nodes of degree at least
    #   k. The coreness of a node is k if the node belongs to the k-core but
    #   not to the (k+1)-core. This function computes k-coreness of all nodes
    #   for a given binary directed connection matrix.
    #
    #   input:          CIJ,        connection/adjacency matrix (binary, directed)
    #
    #   output:    coreness,        node coreness.
    #                    kn,        size of k-core
    #
    #   References: e.g. Hagmann et al. (2008) PLoS Biology
    #
    #   Olaf Sporns, Indiana University, 2007/2008/2010/2012

    N = CIJ.shape[0]  #Get the number of nodes

    coreness = np.zeros(N) # Initialize coreness vector
    kn = np.zeros(N) #Initialize k-core size vector

    for k in range(1, N + 1):
        CIJkcore, kn[k-1] = kcore_bd(CIJ, k) # Assumes kcore_bd is defined elsewhere
        ss = np.sum(CIJkcore, axis=1) > 0 #Find nodes belonging to k-core
        coreness[ss] = k #Assign coreness value k to those nodes

    return coreness, kn

#Placeholder for kcore_bd function.  Replace with actual implementation if available.
def kcore_bd(CIJ,k):
    #This is a placeholder. Replace with the actual implementation of kcore_bd
    #This function should take the connection matrix and k as input and return the k-core matrix and its size.
    #For testing purposes, a dummy implementation is provided.  Replace this with your actual kcore_bd function.
    N = CIJ.shape[0]
    kn = N - k +1 if k <=N else 0
    CIJkcore = np.eye(N)[:kn,:kn] if kn > 0 else np.zeros((N,N))
    return CIJkcore, kn


