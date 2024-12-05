# Translated from degrees_dir.m

import numpy as np

def degrees_dir(CIJ):
    # Indegree and outdegree
    #
    #   [id,od,deg] = degrees_dir(CIJ);
    #
    #   Node degree is the number of links connected to the node. The indegree 
    #   is the number of inward links and the outdegree is the number of 
    #   outward links.
    #
    #   Input:      CIJ,    directed (binary/weighted) connection matrix
    #
    #   Output:     id,     node indegree
    #               od,     node outdegree
    #               deg,    node degree (indegree + outdegree)
    #
    #   Notes:  Inputs are assumed to be on the columns of the CIJ matrix.
    #           Weight information is discarded.
    #
    #
    #   Olaf Sporns, Indiana University, 2002/2006/2008

    # Ensure CIJ is binary...
    CIJ = np.array(CIJ != 0, dtype=float)

    # Compute degrees
    id = np.sum(CIJ, axis=0)  # Indegree = column sum of CIJ
    od = np.sum(CIJ, axis=1)  # Outdegree = row sum of CIJ
    deg = id + od             # Degree = indegree + outdegree
    return id, od, deg


