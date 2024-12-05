# Translated from partition_distance.m

import numpy as np

def partition_distance(Cx, Cy=None):
    """
    Distance or similarity between community partitions.

    This function quantifies information-theoretic distance (normalized
    variation of information) or similarity (normalized mutual information)
    between community partitions.

    Parameters
    ----------
    Cx : array_like
        Community partition vector or matrix of n rows and p columns,
        n is the number of network nodes, and p is the number of input
        community partitions (in the case of vector input p=1).

    Cy : array_like, optional
        Community partition vector or matrix of n rows and q columns. n
        is the number of nodes (must be equal to the number of nodes in
        Cx) and q is the number of input community partitions (may be
        different to the number of nodes in Cx). This argument may be
        omitted, in which case, the partition distance is computed
        between all pairwise partitions of Cx.  Defaults to None.


    Returns
    -------
    VIn : ndarray
        Normalized variation of information ([p, q] matrix)

    MIn : ndarray
        Normalized mutual information ([p, q] matrix)


    Notes
    -----
    Mathematical definitions.

        VIn = [H(X) + H(Y) - 2MI(X, Y)]/log(n)
        MIn = 2MI(X, Y) / [H(X) + H(Y)]

        where H is the entropy and MI is the mutual information


    References
    ----------
    Meila M (2007) J Multivar Anal 98, 873-895.
    """

    s = Cy is None
    if s:
        Cy = Cx
        d = 10**np.ceil(np.log10(1 + np.max(Cx)))
    else:
        d = 10**np.ceil(np.log10(1 + np.max(np.concatenate((Cx.flatten(),Cy.flatten())))))

    if not np.allclose(np.concatenate((Cx.flatten(),Cy.flatten())), np.round(np.concatenate((Cx.flatten(),Cy.flatten()))).astype(int)) or np.min(np.concatenate((Cx.flatten(),Cy.flatten()))) <=0:
        raise ValueError('Input partitions must contain only positive integers.')

    n, p = Cx.shape
    HX = np.zeros((p, 1))
    for i in range(p):
        Px = np.bincount(Cx[:, i]) / n
        Px = Px[Px > 0]  #remove zeros
        HX[i] = -np.sum(Px * np.log(Px))

    if s:
        q = p
        HY = HX
    else:
        n_, q = Cy.shape
        assert n == n_
        HY = np.zeros((q, 1))
        for j in range(q):
            Py = np.bincount(Cy[:, j]) / n
            Py = Py[Py > 0] #remove zeros
            HY[j] = -np.sum(Py * np.log(Py))

    VIn = np.zeros((p, q))
    MIn = np.zeros((p, q))
    for i in range(p):
        if s:
            j_idx = range(i,q)
        else:
            j_idx = range(q)
        for j in j_idx:
            Pxy = np.bincount(d * Cx[:, i] + Cy[:, j]) / n
            Pxy = Pxy[Pxy > 0] #remove zeros
            Hxy = -np.sum(Pxy * np.log(Pxy))
            VIn[i, j] = (2 * Hxy - HX[i] - HY[j]) / np.log(n)
            MIn[i, j] = 2 * (HX[i] + HY[j] - Hxy) / (HX[i] + HY[j])
        if s:
            VIn[j_idx, i] = VIn[i, j_idx]
            MIn[j_idx, i] = MIn[i, j_idx]

    return VIn, MIn

