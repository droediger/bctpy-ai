# Translated from get_components.m

import numpy as np
from scipy.sparse.csgraph import dmperm

def get_components(adj):
    """
    Connected components

    Parameters
    ----------
    adj : numpy.ndarray
        Binary and undirected adjacency matrix

    Returns
    -------
    comps : numpy.ndarray
        Vector of component assignments for each node
    comp_sizes : numpy.ndarray
        Vector of component sizes

    Notes
    -----
    Disconnected nodes will appear as components of size 1
    """

    # Check if the adjacency matrix is square
    if adj.shape[0] != adj.shape[1]:
        raise ValueError('This adjacency matrix is not square')

    # Ensure the adjacency matrix is symmetric (undirected graph)
    adj = np.maximum(adj, adj.T)

    # Add self-loops if necessary (ensure diagonal contains all ones)
    adj = np.maximum(adj, np.eye(adj.shape[0]))

    # Dulmage-Mendelsohn decomposition
    _, p, _, r = dmperm(adj)

    # Component sizes: difference between consecutive component boundaries
    comp_sizes = np.diff(r)

    # Number of components
    num_comps = len(comp_sizes)

    # Initialize component assignments
    comps = np.zeros(adj.shape[0], dtype=int)

    # Assign component labels
    comps[r[:num_comps]] = np.arange(1, num_comps + 1)
    comps = np.cumsum(comps)

    # Reorder component labels according to the permutation
    comps = comps[p]

    return comps, comp_sizes


