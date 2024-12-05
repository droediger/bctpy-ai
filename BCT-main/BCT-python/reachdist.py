# Translated from reachdist.m

import numpy as np

def reachdist(CIJ):
    """
    Reachability and distance matrices

    Parameters
    ----------
    CIJ : numpy.ndarray
        Binary (directed/undirected) connection matrix

    Returns
    -------
    R : numpy.ndarray
        Reachability matrix
    D : numpy.ndarray
        Distance matrix

    Notes
    -----
    Faster but more memory intensive than "breadthdist.m".
    Algorithm: algebraic path count.
    """

    # Initialize
    R = CIJ.copy()
    D = CIJ.copy()
    powr = 2
    N = CIJ.shape[0]
    CIJpwr = CIJ.copy()

    # Check for vertices with no incoming or outgoing connections
    id = np.sum(CIJ, axis=1)  # Indegree = column sum of CIJ
    od = np.sum(CIJ, axis=0)  # Outdegree = row sum of CIJ
    id_0 = np.where(id == 0)[0]  # Nothing goes in, so column(R) will be 0
    od_0 = np.where(od == 0)[0]  # Nothing comes out, so row(R) will be 0
    # Use these columns and rows to check for reachability:
    col = np.setdiff1d(np.arange(N), id_0)
    row = np.setdiff1d(np.arange(N), od_0)

    R, D, powr = reachdist2(CIJ, CIJpwr, R, D, N, powr, col, row)

    # "Invert" CIJdist to get distances
    D = powr - D + 1

    # Put 'Inf' if no path found
    D[D == (N + 2)] = np.inf
    D[:, id_0] = np.inf
    D[od_0, :] = np.inf

    return R, D


def reachdist2(CIJ, CIJpwr, R, D, N, powr, col, row):
    """
    Recursive helper function for reachdist

    Parameters
    ----------
    CIJ : numpy.ndarray
        Binary (directed/undirected) connection matrix
    CIJpwr : numpy.ndarray
        Current power of CIJ
    R : numpy.ndarray
        Reachability matrix
    D : numpy.ndarray
        Distance matrix
    N : int
        Number of nodes
    powr : int
        Current power
    col : numpy.ndarray
        Columns to consider for reachability
    row : numpy.ndarray
        Rows to consider for reachability

    Returns
    -------
    R : numpy.ndarray
        Updated reachability matrix
    D : numpy.ndarray
        Updated distance matrix
    powr : int
        Updated power
    """

    CIJpwr = CIJpwr @ CIJ
    R = R | (CIJpwr != 0).astype(int) #Use of double is unnecessary in Python
    D = D + R

    if (powr <= N) and np.any(R[np.ix_(row, col)] == 0):
        powr = powr + 1
        R, D, powr = reachdist2(CIJ, CIJpwr, R, D, N, powr, col, row)

    return R, D, powr

