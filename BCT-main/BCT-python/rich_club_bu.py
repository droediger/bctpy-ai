# Translated from rich_club_bu.m

import numpy as np

def rich_club_bu(CIJ, klevel=None):
    """
    Rich club coefficients (binary undirected graph)

    Parameters
    ----------
    CIJ : array_like
        Connection matrix, binary and undirected.
    klevel : int, optional
        Maximum level at which the rich club coefficient will be calculated. 
        If not included, the maximum level will be set to the maximum degree of CIJ.

    Returns
    -------
    R : ndarray
        Vector of rich-club coefficients for levels 1 to klevel.
    Nk : ndarray
        Number of nodes with degree > k.
    Ek : ndarray
        Number of edges remaining in subgraph with degree > k.

    References
    ----------
    Colizza et al. (2006) Nat. Phys. 2:110.
    """

    Degree = np.sum(CIJ, axis=0)  # Compute degree of each node

    if klevel is None:
        klevel = np.max(Degree)
    elif not isinstance(klevel, int) or klevel <=0:
        raise ValueError("klevel must be a positive integer")

    R = np.zeros(klevel)
    Nk = np.zeros(klevel)
    Ek = np.zeros(klevel)

    for k in range(1, klevel + 1):
        SmallNodes = np.where(Degree <= k)[0]  # Get 'small nodes' with degree <= k
        subCIJ = np.delete(np.delete(CIJ, SmallNodes, axis=0), SmallNodes, axis=1) # Extract subnetwork of nodes > k by removing nodes <= k from CIJ
        Nk[k-1] = subCIJ.shape[1]  # Number of nodes with degree > k
        Ek[k-1] = np.sum(subCIJ)  # Total number of connections in subgraph
        if Nk[k-1] <=1:
            R[k-1] = 0 #handle cases where Nk is 0 or 1 to avoid division by zero
        else:
            R[k-1] = Ek[k-1] / (Nk[k-1] * (Nk[k-1] - 1))  # Unweighted rich-club coefficient

    return R, Nk, Ek

