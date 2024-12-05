# Translated from assortativity_wei.m

import numpy as np

def assortativity_wei(CIJ, flag):
    """
    Assortativity coefficient.

    The assortativity coefficient is a correlation coefficient between the
    strengths (weighted degrees) of all nodes on two opposite ends of a link.
    A positive assortativity coefficient indicates that nodes tend to link to
    other nodes with the same or similar strength.

    Args:
        CIJ (numpy.ndarray): Weighted directed/undirected connection matrix.
        flag (int): 
            0: undirected graph: strength/strength correlation
            1: directed graph: out-strength/in-strength correlation
            2: directed graph: in-strength/out-strength correlation
            3: directed graph: out-strength/out-strength correlation
            4: directed graph: in-strength/in-strength correlation

    Returns:
        float: Assortativity coefficient.

    Notes:
        The main diagonal should be empty. For flag 1 the function computes
        the directed assortativity described in Rubinov and Sporns (2010) NeuroImage.

    References:
        Newman (2002) Phys Rev Lett 89:208701
        Foster et al. (2010) PNAS 107:10815-10820
    """

    if flag == 0:  # Undirected version
        str = strengths_und(CIJ) # Assume strengths_und is defined elsewhere
        i, j = np.where(np.triu(CIJ, 1) > 0)
        K = len(i)
        stri = str[i]
        strj = str[j]
    else:  # Directed versions
        is_, os_ = strengths_dir(CIJ) # Assume strengths_dir is defined elsewhere
        i, j = np.where(CIJ > 0)
        K = len(i)

        if flag == 1:
            stri = os_[i]
            strj = is_[j]
        elif flag == 2:
            stri = is_[i]
            strj = os_[j]
        elif flag == 3:
            stri = os_[i]
            strj = os_[j]
        elif flag == 4:
            stri = is_[i]
            strj = is_[j]

    # Compute assortativity
    r = (np.sum(stri * strj) / K - (np.sum(0.5 * (stri + strj)) / K)**2) / \
        (np.sum(0.5 * (stri**2 + strj**2)) / K - (np.sum(0.5 * (stri + strj)) / K)**2)

    return r


