# Translated from edge_nei_overlap_bu.m

import numpy as np

def edge_nei_overlap_bu(CIJ):
    # EDGE_NEI_OVERLAP_BU        Overlap amongst neighbors of two adjacent nodes
    #
    #   [EC,ec,degij] = edge_nei_bu(CIJ);
    #
    #   This function determines the neighbors of two nodes that are linked by 
    #   an edge, and then computes their overlap.  Connection matrix must be
    #   binary and directed.  Entries of 'EC' that are 'inf' indicate that no
    #   edge is present.  Entries of 'EC' that are 0 denote "local bridges", i.e.
    #   edges that link completely non-overlapping neighborhoods.  Low values
    #   of EC indicate edges that are "weak ties".
    #
    #   If CIJ is weighted, the weights are ignored.
    #
    #   Inputs:     CIJ,    undirected (binary/weighted) connection matrix
    #  
    #   Outputs:    EC,     edge neighborhood overlap matrix
    #               ec,     edge neighborhood overlap per edge, in vector format
    #               degij,  degrees of node pairs connected by each edge
    #
    #   Reference: Easley and Kleinberg (2010) Networks, Crowds, and Markets. 
    #              Cambridge University Press, Chapter 3.
    #
    #   Olaf Sporns, Indiana University, 2012

    CIJ = np.asarray(CIJ, dtype=bool) # Ensure CIJ is boolean for binary operations

    ik, jk, ck = np.where(CIJ)
    lel = len(ck)
    N = CIJ.shape[0]

    deg = np.sum(CIJ, axis=0) + np.sum(CIJ, axis=1) # degree of each node (undirected)

    ec = np.zeros(lel)
    degij = np.zeros((2, lel))
    for e in range(lel):
        neiik = np.setdiff1d(np.union1d(np.where(CIJ[ik[e],:])[0], np.where(CIJ[:,ik[e]])[0]), [ik[e], jk[e]])
        neijk = np.setdiff1d(np.union1d(np.where(CIJ[jk[e],:])[0], np.where(CIJ[:,jk[e]])[0]), [ik[e], jk[e]])
        ec[e] = len(np.intersect1d(neiik, neijk)) / len(np.union1d(neiik, neijk)) if len(np.union1d(neiik, neijk)) > 0 else 0 #handle empty union case
        degij[:, e] = [deg[ik[e]], deg[jk[e]]]

    ff = np.where(CIJ)
    EC = np.full((N,N), np.inf) # Initialize EC with inf to represent missing edges
    EC[ff] = ec

    return EC, ec, degij

