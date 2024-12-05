# Translated from assortativity_bin.m

import numpy as np

def assortativity_bin(CIJ, flag):
    """
    Assortativity coefficient

    Computes the assortativity coefficient, a correlation coefficient between the degrees of nodes at opposite ends of a link.  A positive coefficient indicates that nodes tend to link to others with similar degrees.

    Args:
        CIJ (numpy.ndarray): Binary directed/undirected connection matrix.
        flag (int):  Specifies the type of assortativity to compute:
                     0: Undirected graph (degree/degree correlation)
                     1: Directed graph (out-degree/in-degree correlation)
                     2: Directed graph (in-degree/out-degree correlation)
                     3: Directed graph (out-degree/out-degree correlation)
                     4: Directed graph (in-degree/in-degree correlation)

    Returns:
        float: Assortativity coefficient (r).

    Notes:
        Accepts weighted networks, but connection weights are ignored. The main diagonal of CIJ should be empty.  For flag=1, computes the directed assortativity as described in Rubinov and Sporns (2010) NeuroImage.

    References:
        Newman (2002) Phys Rev Lett 89:208701
        Foster et al. (2010) PNAS 107:10815-10820
    """

    if flag == 0:  # Undirected version
        deg = degrees_und(CIJ)  # Assumed to be defined elsewhere
        i, j = np.nonzero(np.triu(CIJ, 1))
        K = len(i)
        degi = deg[i]
        degj = deg[j]
    else:  # Directed versions
        id, od = degrees_dir(CIJ)  # Assumed to be defined elsewhere
        i, j = np.nonzero(CIJ)
        K = len(i)

        if flag == 1:
            degi = od[i]
            degj = id[j]
        elif flag == 2:
            degi = id[i]
            degj = od[j]
        elif flag == 3:
            degi = od[i]
            degj = od[j]
        elif flag == 4:
            degi = id[i]
            degj = id[j]

    #Compute assortativity
    r = (np.sum(degi * degj) / K - (np.sum(0.5 * (degi + degj)) / K)**2) / \
        (np.sum(0.5 * (degi**2 + degj**2)) / K - (np.sum(0.5 * (degi + degj)) / K)**2)

    return r


