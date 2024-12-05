# Translated from edge_nei_overlap_bd.m

import numpy as np

def edge_nei_overlap_bd(CIJ):
    # EDGE_NEI_OVERLAP_BD     Overlap amongst neighbors of two adjacent nodes
    #
    #   [EC,ec,degij] = edge_nei_bd(CIJ);
    #
    #   This function determines the neighbors of two nodes that are linked by 
    #   an edge, and then computes their overlap.  Connection matrix must be
    #   binary and directed.  Entries of 'EC' that are 'inf' indicate that no
    #   edge is present.  Entries of 'EC' that are 0 denote "local bridges",
    #   i.e. edges that link completely non-overlapping neighborhoods.  Low
    #   values of EC indicate edges that are "weak ties".
    #
    #   If CIJ is weighted, the weights are ignored. Neighbors of a node can be
    #   linked by incoming, outgoing, or reciprocal connections.
    #
    #   Inputs:     CIJ,      directed (binary/weighted) connection matrix
    #  
    #   Outputs:    EC,     edge neighborhood overlap matrix
    #               ec,     edge neighborhood overlap per edge, in vector format
    #               degij,  degrees of node pairs connected by each edge
    #
    #   Reference:
    #
    #       Easley and Kleinberg (2010) Networks, Crowds, and Markets. 
    #           Cambridge University Press, Chapter 3
    #
    #   Olaf Sporns, Indiana University, 2012

    ik, jk, ck = np.nonzero(CIJ)
    lel = len(ck)
    N = CIJ.shape[0]

    # Assuming degrees_dir function is defined elsewhere
    deg = degrees_dir(CIJ)[2]

    ec = np.zeros(lel)
    degij = np.zeros((2, lel))
    for e in range(lel):
        neiik = np.setdiff1d(np.union1d(np.nonzero(CIJ[ik[e],:])[0], np.nonzero(CIJ[:,ik[e]])[0]), [ik[e], jk[e]])
        neijk = np.setdiff1d(np.union1d(np.nonzero(CIJ[jk[e],:])[0], np.nonzero(CIJ[:,jk[e]])[0]), [ik[e], jk[e]])
        ec[e] = len(np.intersect1d(neiik, neijk)) / len(np.union1d(neiik, neijk)) if len(np.union1d(neiik, neijk)) > 0 else 0 #Handle empty union case.
        degij[:, e] = [deg[ik[e]], deg[jk[e]]]

    ff = np.nonzero(CIJ)
    EC = np.full((N,N), np.inf) # Initialize with inf, representing no edge
    EC[ff] = ec

    return EC, ec, degij

# Placeholder for degrees_dir function.  Replace with your actual implementation.
def degrees_dir(CIJ):
    #This is a placeholder, replace with your actual implementation
    in_degree = np.sum(CIJ, axis=0)
    out_degree = np.sum(CIJ, axis=1)
    return in_degree, out_degree, in_degree + out_degree

