# Translated from gateway_coef_sign.m

import numpy as np

def gateway_coef_sign(W, Ci, centtype):
    """
    Gateway coefficient

    Parameters
    ----------
    W : numpy.ndarray
        Undirected connection matrix with positive and negative weights.
    Ci : numpy.ndarray
        Community affiliation vector.
    centtype : int
        Centrality measure to use:
            1 = Node Strength
            2 = Betweenness Centrality

    Returns
    -------
    GWpos : numpy.ndarray
        Gateway coefficient for positive weights.
    GWneg : numpy.ndarray
        Gateway coefficient for negative weights.

    References
    ----------
    Vargas ER, Wahl LM. Eur Phys J B (2014) 87:1-10.
    """

    _, idx, Ci = np.unique(Ci, return_index=True, return_inverse=True) # Remap module indices to consecutive numbers
    n = len(W)  # Number of nodes
    np.fill_diagonal(W, 0)  # Ensure diagonal is zero
    GWpos = gcoef(W * (W > 0), Ci, centtype)  # Compute gateway coefficient for positive weights
    GWneg = gcoef(-W * (W < 0), Ci, centtype)  # Compute gateway coefficient for negative weights
    return GWpos, GWneg


def gcoef(W_, Ci, centtype):
    """
    Compute gateway coefficient.

    Parameters
    ----------
    W_ : numpy.ndarray
        Weighted adjacency matrix.
    Ci : numpy.ndarray
        Community affiliation vector.
    centtype : int
        Centrality measure to use.

    Returns
    -------
    GW : numpy.ndarray
        Gateway coefficient.
    """
    k = np.sum(W_, axis=1)  # Compute node strength
    n = len(W_)
    Gc = (W_ != 0) * Ci[:, np.newaxis] # Create neighbor community affiliation matrix
    nmod = np.max(Ci) + 1 # Find # of modules
    ks = np.zeros((n, nmod))  # Preallocate space
    kjs = np.zeros((n, nmod))  # Preallocate space
    cs = np.zeros((n, nmod))  # Preallocate space

    if centtype == 1:  # Node Strength
        cent = np.sum(W_, axis=1)
    elif centtype == 2:  # Betweenness Centrality
        # Assuming weight_conversion and betweenness_wei are defined elsewhere
        L = weight_conversion(W_, 'lengths')
        cent = betweenness_wei(L)
    else:
        raise ValueError("Invalid centtype.")

    mcn = 0  # Set max summed centrality per module to 0
    for i in range(nmod):  # For each module
        if np.sum(cent[Ci == i]) > mcn:  # If current module has a higher sum
            mcn = np.sum(cent[Ci == i])  # Reassign value
        ks[:, i] = np.sum(W_ * (Gc == i), axis=1)  # Compute the total weight of the connections per node to each module

    for i in range(nmod):  # For each module
        if np.sum(Ci == i) > 1:  # If there is more than 1 node in a module
            kjs[Ci == i, :] = np.tile(np.sum(ks[Ci == i, :], axis=0), (np.sum(Ci == i), 1))  # Compute total module-module connections
            kjs[Ci == i, i] = kjs[Ci == i, i] / 2  # Account for redundancy due to double counting within-network work weights

    for i in range(n):  # For each node
        if k[i] > 0:  # If node is connected
            for ii in range(nmod):  # For each module
                cs[i, ii] = np.sum(cent[(Ci * (W_[:, i] > 0)) == ii])  # Sum of centralities of neighbors of a node within a module

    ksm = ks / kjs  # Normalize by total connections
    ksm[kjs == 0] = 0  # Account for division by 0
    csm = cs / mcn  # Normalize by max summed centrality
    gs = (1 - (ksm * csm)) ** 2  # Calculate total weighting
    GW = 1 - np.sum((ks ** 2) / (k[:, np.newaxis] ** 2) * gs, axis=1)  # Compute gateway coefficient
    GW[np.isnan(GW)] = 0  # Account for division by 0
    GW[GW == 0] = 0  # Set to 0 if no neighbors
    return GW


# Placeholder functions - replace with actual implementations if needed
def weight_conversion(W,type):
    return W

def betweenness_wei(W):
    return np.zeros(len(W))

