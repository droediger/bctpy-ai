# Translated from rich_club_bd.m

import numpy as np

def rich_club_bd(CIJ, klevel=None):
    """
    Rich club coefficients (binary directed graph)

    Parameters
    ----------
    CIJ : numpy.ndarray
        Connection matrix, binary and directed.
    klevel : int, optional
        Maximum level at which the rich club coefficient will be calculated. 
        If not included, the maximum level will be set to the maximum degree of CIJ.

    Returns
    -------
    R : numpy.ndarray
        Vector of rich-club coefficients for levels 1 to klevel.
    Nk : numpy.ndarray
        Number of nodes with degree > k.
    Ek : numpy.ndarray
        Number of edges remaining in subgraph with degree > k.

    Note
    ----
    This function assumes the existence of a degrees_dir function which is not defined here but is assumed to be available elsewhere.  It should take a connection matrix as input and return the in-degree, out-degree and total degree of each node.

    """
    N = CIJ.shape[0]  # Number of nodes

    # Definition of "degree" as used for RC coefficients. Degree is the sum of incoming and outgoing connections.
    in_degree, out_degree, degree = degrees_dir(CIJ) #Assumed to be defined elsewhere

    if klevel is None:
        klevel = np.max(degree)
    elif not isinstance(klevel, int) or klevel <=0:
        raise ValueError("klevel must be a positive integer")


    R = np.zeros(klevel)
    Nk = np.zeros(klevel)
    Ek = np.zeros(klevel)

    for k in range(1, klevel + 1):
        SmallNodes = np.where(degree <= k)[0]  # Get 'small nodes' with degree <= k
        subCIJ = np.delete(np.delete(CIJ, SmallNodes, axis=0), SmallNodes, axis=1)  # Extract subnetwork of nodes > k
        Nk[k - 1] = subCIJ.shape[1]  # Number of nodes with degree > k
        Ek[k - 1] = np.sum(subCIJ)  # Total number of connections in subgraph
        if Nk[k-1] > 1:
            R[k - 1] = Ek[k - 1] / (Nk[k - 1] * (Nk[k - 1] - 1))  # Unweighted rich-club coefficient
        else:
            R[k-1] = 0 #Handle case where Nk(k) <=1 to avoid division by zero

    return R, Nk, Ek

#Dummy degrees_dir function for testing purposes. Replace with your actual implementation.
def degrees_dir(CIJ):
    in_degree = np.sum(CIJ, axis=0)
    out_degree = np.sum(CIJ, axis=1)
    degree = in_degree + out_degree
    return in_degree, out_degree, degree

