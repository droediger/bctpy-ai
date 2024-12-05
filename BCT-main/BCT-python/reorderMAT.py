# Translated from reorderMAT.m

import numpy as np
from scipy.stats import norm

def reorderMAT(MAT, H, cost):
    """Reorder matrix for visualization

    Args:
        MAT (numpy.ndarray): connection matrix
        H (int): number of reordering attempts
        cost (str): 'line' or 'circ', for shape of lattice (linear or ring lattice)

    Returns:
        tuple: reordered connection matrix, reordered indices, cost of reordered matrix
    """

    N = len(MAT)
    diagMAT = np.diag(np.diag(MAT))
    MAT = MAT - diagMAT

    # generate cost function
    if cost == 'line':
        profil = np.flip(norm.pdf(np.arange(1, N + 1), 0, N / 2))
    elif cost == 'circ':
        profil = np.flip(norm.pdf(np.arange(1, N + 1), N / 2, N / 4))
    else:
        raise ValueError("cost must be 'line' or 'circ'")

    COST = np.tile(profil, (N, 1)) * np.tile(profil[:,None], (1,N))


    # initialize lowCOST
    lowMATcost = np.sum(COST * MAT)

    # keep track of starting configuration
    MATstart = MAT.copy()
    starta = np.arange(1, N + 1)

    # reorder
    for h in range(H):
        a = np.arange(1, N + 1)
        # choose two positions at random and flip them
        r = np.random.permutation(N)
        a[r[0] -1], a[r[1] -1] = a[r[1] -1], a[r[0] -1]
        MATcostnew = np.sum(MAT[a - 1, a - 1] * COST)
        if MATcostnew < lowMATcost:
            MAT = MAT[a - 1, a - 1]
            starta[r[0]-1], starta[r[1]-1] = starta[r[1]-1], starta[r[0]-1]
            lowMATcost = MATcostnew

    MATreordered = MATstart[starta - 1, starta - 1] + diagMAT[starta - 1, starta - 1]
    MATindices = starta
    MATcost = lowMATcost

    return MATreordered, MATindices, MATcost

