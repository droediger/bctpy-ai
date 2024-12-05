# Translated from score_wu.m

import numpy as np

def score_wu(CIJ, s):
    """
    Computes the s-core for a given weighted undirected connection matrix.

    The s-core is the largest subnetwork comprising nodes of strength at 
    least s.  Computation is analogous to the more widely used k-core, but 
    is based on node strengths instead of node degrees.

    Args:
        CIJ (numpy.ndarray): Connection/adjacency matrix (weighted, undirected).
        s (float): Level of s-core.  Note: s can take on any fractional value.

    Returns:
        tuple: CIJscore (numpy.ndarray): Connection matrix of the s-core. This 
               matrix contains only nodes with a strength of at least s.
               sn (int): Size of s-core.
    """

    def strengths_und(CIJ):
        # Assume strengths_und function is defined elsewhere
        return np.sum(CIJ, axis=0)

    while True:
        # Get strengths of matrix
        str = strengths_und(CIJ)

        # Find nodes with strength < s
        ff = np.where((str < s) & (str > 0))[0]

        # If none found -> stop
        if len(ff) == 0:
            break

        # Peel found nodes
        CIJ[ff, :] = 0
        CIJ[:, ff] = 0

    CIJscore = CIJ
    sn = np.sum(str > 0)
    return CIJscore, sn



