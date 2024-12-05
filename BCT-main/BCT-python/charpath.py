# Translated from charpath.m

import numpy as np

def charpath(D, diagonal_dist=0, infinite_dist=1):
    """
    Characteristic path length, global efficiency and related statistics

    Parameters
    ----------
    D : array_like
        Distance matrix.
    diagonal_dist : bool, optional
        Include distances on the main diagonal (default: 0).
    infinite_dist : bool, optional
        Include infinite distances in calculation (default: 1).

    Returns
    -------
    lambda : float
        Network characteristic path length.
    efficiency : float
        Network global efficiency.
    ecc : array_like
        Nodal eccentricity.
    radius : float
        Network radius.
    diameter : float
        Network diameter.

    Notes
    -----
    The input distance matrix may be obtained with any of the distance
    functions, e.g. distance_bin, distance_wei.
    Characteristic path length is defined here as the mean shortest
    path length between all pairs of nodes, for consistency with common
    usage. Note that characteristic path length is also defined as the
    median of the mean shortest path length from each node to all other
    nodes.
    Infinitely long paths (i.e. paths between disconnected nodes) are
    included in computations by default. This behavior may be modified with
    via the infinite_dist argument.
    """

    n = D.shape[0]
    if np.isnan(D).any():
        raise ValueError('The distance matrix must not contain NaN values')
    if not diagonal_dist:
        np.fill_diagonal(D, np.nan)  # Set diagonal distance to NaN
    if not infinite_dist:
        D[np.isinf(D)] = np.nan  # Ignore infinite path lengths

    Dv = D[~np.isnan(D)]  # Get non-NaN indices of D

    # Mean of entries of D(G)
    lambda_ = np.mean(Dv)

    # Efficiency: mean of inverse entries of D(G)
    efficiency = np.mean(1. / Dv)

    # Eccentricity for each vertex
    ecc = np.nanmax(D, axis=1)

    # Radius of graph
    radius = np.min(ecc)

    # Diameter of graph
    diameter = np.max(ecc)

    return lambda_, efficiency, ecc, radius, diameter

