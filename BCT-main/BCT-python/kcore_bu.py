# Translated from kcore_bu.m

import numpy as np

def kcore_bu(CIJ, k):
    # K-core decomposition
    #
    #   [CIJkcore,kn,peelorder,peellevel] = kcore_bu(CIJ,k);
    #
    #   The k-core is the largest subnetwork comprising nodes of degree at
    #   least k. This function computes the k-core for a given binary
    #   undirected connection matrix by recursively peeling off nodes with
    #   degree lower than k, until no such nodes remain.
    #
    #   input:          CIJ,        connection/adjacency matrix (binary, undirected)
    #                     k,        level of k-core
    #
    #   output:    CIJkcore,        connection matrix of the k-core.  This matrix
    #                               only contains nodes of degree at least k.
    #                    kn,        size of k-core
    #                    peelorder, indices in the order in which they were
    #                               peeled away during k-core decomposition
    #                    peellevel, corresponding level - nodes at the same
    #                               level were peeled away at the same time
    #
    #   'peelorder' and 'peellevel' are similar the the k-core sub-shells
    #   described in Modha and Singh (2010).
    #
    #   References: e.g. Hagmann et al. (2008) PLoS Biology
    #
    #   Olaf Sporns, Indiana University, 2007/2008/2010/2012

    peelorder = np.array([])
    peellevel = np.array([])
    iter = 0

    while True:
        # get degrees of matrix
        deg = np.sum(CIJ, axis=0)

        # find nodes with degree <k
        ff = np.where((deg < k) & (deg > 0))[0]

        # if none found -> stop
        if len(ff) == 0:
            break

        # peel away found nodes
        iter += 1
        CIJ[ff, :] = 0
        CIJ[:, ff] = 0

        peelorder = np.concatenate((peelorder, ff))
        peellevel = np.concatenate((peellevel, np.repeat(iter, len(ff))))

    CIJkcore = CIJ
    kn = np.sum(deg > 0)

    return CIJkcore, kn, peelorder, peellevel

def degrees_und(CIJ):
    #This function is assumed to be defined elsewhere and computes the degree of each node in an undirected graph represented by the adjacency matrix CIJ
    deg = np.sum(CIJ, axis=0)
    return deg

