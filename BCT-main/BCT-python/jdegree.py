# Translated from jdegree.m

import numpy as np

def jdegree(CIJ):
    """Joint degree distribution

    Args:
        CIJ (numpy.ndarray): directed (weighted/binary) connection matrix

    Returns:
        tuple: J (numpy.ndarray): joint degree distribution matrix (shifted by one)
               J_od (int): number of vertices with od > id.
               J_id (int): number of vertices with id > od.
               J_bl (int): number of vertices with id = od.

    Notes:
        Weights are discarded.
    """

    # Ensure CIJ is binary...
    CIJ = (CIJ != 0).astype(int)

    N = CIJ.shape[0]

    id = np.sum(CIJ, axis=0)  # Indegree = column sum of CIJ
    od = np.sum(CIJ, axis=1)  # Outdegree = row sum of CIJ

    # Create the joint degree distribution matrix
    # Note: The matrix is shifted by one, to accommodate zero id and od in the first row/column.
    # Upper triangular part of the matrix has vertices with an excess of outgoing edges (od > id)
    # Lower triangular part of the matrix has vertices with an excess of incoming edges (id > od)
    # Main diagonal has units with id = od

    szJ = np.max(np.maximum(id, od)) + 1
    J = np.zeros((szJ, szJ), dtype=int)

    for i in range(N):
        J[id[i], od[i]] += 1

    J_od = np.sum(np.triu(J, 1))
    J_id = np.sum(np.tril(J, -1))
    J_bl = np.sum(np.diag(J))

    return J, J_od, J_id, J_bl

