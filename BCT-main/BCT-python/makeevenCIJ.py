# Translated from makeevenCIJ.m

import numpy as np

def makeevenCIJ(N, K, sz_cl):
    """
    Generates a random, directed network with a specified number of fully 
    connected modules linked together by evenly distributed remaining random 
    connections.

    Args:
        N: Number of vertices (must be a power of 2).
        K: Number of edges.
        sz_cl: Size of clusters (power of 2).

    Returns:
        CIJ: Connection matrix.
    """

    # Compute number of hierarchical levels and adjust cluster size
    mx_lvl = int(np.floor(np.log2(N)))
    sz_cl -= 1

    # Create a small template
    t = np.ones((2, 2)) * 2

    # Check N against the number of levels
    Nlvl = 2**mx_lvl
    if Nlvl != N:
        print('Warning: N must be a power of 2')
    N = Nlvl

    # Create hierarchical template
    CIJ = np.ones((2**(mx_lvl),2**(mx_lvl)))
    for lvl in range(1, mx_lvl):
        CIJ = np.ones((2**(lvl+1),2**(lvl+1)))
        group1 = np.arange(2**lvl)
        group2 = np.arange(2**lvl, 2**(lvl+1))
        CIJ[np.ix_(group1,group1)] = t
        CIJ[np.ix_(group2,group2)] = t
        CIJ += 1
        t = CIJ

    s = CIJ.shape[0]
    CIJ = CIJ - 1 - mx_lvl * np.eye(s)

    # Assign connection probabilities
    CIJp = CIJ >= (mx_lvl - sz_cl)

    # Determine number of remaining (non-cluster) connections and their
    # possible positions
    CIJc = CJI = (CIJp == 1)
    remK = K - np.count_nonzero(CIJc)
    if remK < 0:
        print('Warning: K is too small, output matrix contains clusters only')

    a, b = np.nonzero(~(CIJc + np.eye(N)))

    # Assign 'remK' randomly distributed connections
    rp = np.random.permutation(len(a))
    a = a[rp[:remK]]
    b = b[rp[:remK]]
    for i in range(remK):
        CIJc[a[i], b[i]] = 1

    # Prepare for output
    CIJ = CIJc

    return CIJ

