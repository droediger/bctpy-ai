# Translated from makefractalCIJ.m

import numpy as np

def makefractalCIJ(mx_lvl, E, sz_cl):
    """
    Generates a directed network with a hierarchical modular organization.

    Args:
        mx_lvl: Number of hierarchical levels, N = 2^mx_lvl.
        E: Connection density fall-off per level.
        sz_cl: Size of clusters (power of 2).

    Returns:
        CIJ: Connection matrix.
        K: Number of connections present in the output CIJ.
    """

    # Create a 2x2 template matrix
    t = 2 * np.ones((2, 2))

    # Compute N and adjust cluster size
    N = 2**mx_lvl
    sz_cl -= 1

    # Iterate through hierarchical levels
    for lvl in range(1, mx_lvl):
        CIJ = np.ones((2**(lvl + 1), 2**(lvl + 1)))
        group1 = np.arange(2**lvl)
        group2 = np.arange(2**lvl, 2**(lvl+1))
        CIJ[np.ix_(group1, group1)] = t
        CIJ[np.ix_(group2, group2)] = t
        CIJ += np.ones((CIJ.shape[0], CIJ.shape[1]))
        t = CIJ

    s = CIJ.shape[0]
    CIJ = CIJ - np.ones((s, s)) - mx_lvl * np.eye(s)

    # Assign connection probabilities
    ee = mx_lvl - CIJ - sz_cl
    ee = np.maximum(ee, 0) # equivalent to (ee>0).*ee in MATLAB
    prob = (1 / (E**ee)) * (np.ones((s, s)) - np.eye(s))
    CIJ = (prob > np.random.rand(N, N)).astype(int) #astype(int) converts boolean to 0/1

    # Count connections
    K = np.sum(CIJ)

    return CIJ, K

