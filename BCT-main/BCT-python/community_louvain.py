# Translated from community_louvain.m

import numpy as np

def community_louvain(W, gamma=1, M0=None, B='modularity'):
    """
    Optimal community structure

    Parameters
    ----------
    W : array_like
        Directed/undirected weighted/binary connection matrix with positive and possibly negative weights.
    gamma : float, optional
        Resolution parameter (optional)
            gamma>1,        detects smaller modules
            0<=gamma<1,     detects larger modules
            gamma=1,        classic modularity (default)
    M0 : array_like, optional
        Initial community affiliation vector (optional)
    B : str or array_like, optional
        Objective-function type or custom objective matrix (optional)
        'modularity',       modularity (default)
        'potts',            Potts-model Hamiltonian (for binary networks)
        'negative_sym',     symmetric treatment of negative weights
        'negative_asym',    asymmetric treatment of negative weights
        B,                  custom objective-function matrix

        Note: see Rubinov and Sporns (2011) for a discussion of
        symmetric vs. asymmetric treatment of negative weights.

    Returns
    -------
    M : array_like
        Community affiliation vector
    Q : float
        Optimized community-structure statistic (modularity by default)
    """

    W = np.array(W, dtype=float)  # convert to double format
    n = len(W)  # get number of nodes
    s = np.sum(W)  # get sum of edges

    if B is None or B == 'modularity':
        type_B = 'modularity'
    elif isinstance(B, str):
        type_B = B
    else:
        type_B = 0
        if gamma is not None:
            print('Warning: Value of gamma is ignored in generalized mode.')

    if gamma is None:
        gamma = 1

    if type_B == 'negative_sym' or type_B == 'negative_asym':
        W0 = W * (W > 0)  # positive weights matrix
        s0 = np.sum(W0)  # weight of positive links
        B0 = W0 - gamma * (np.sum(W0, axis=1, keepdims=True) @ np.sum(W0, axis=0, keepdims=True)) / s0  # positive modularity

        W1 = -W * (W < 0)  # negative weights matrix
        s1 = np.sum(W1)  # weight of negative links
        if s1:  # negative modularity
            B1 = W1 - gamma * (np.sum(W1, axis=1, keepdims=True) @ np.sum(W1, axis=0, keepdims=True)) / s1
        else:
            B1 = 0
    elif np.min(W) < -1e-10:
        raise ValueError('The input connection matrix contains negative weights.\nSpecify \'negative_sym\' or \'negative_asym\' objective-function types.')
    
    if type_B == 'potts' and np.any(W != W.astype(bool)):
        raise ValueError('Potts-model Hamiltonian requires a binary W.')

    if type_B:
        if type_B == 'modularity':
            B = (W - gamma * (np.sum(W, axis=1, keepdims=True) @ np.sum(W, axis=0, keepdims=True)) / s) / s
        elif type_B == 'potts':
            B = W - gamma * (1 - W)
        elif type_B == 'negative_sym':
            B = B0 / (s0 + s1) - B1 / (s0 + s1)
        elif type_B == 'negative_asym':
            B = B0 / s0 - B1 / (s0 + s1)
        else:
            raise ValueError('Unknown objective function.')
    else:  # custom objective function matrix as input
        B = np.array(B, dtype=float)
        if not np.array_equal(W.shape, B.shape):
            raise ValueError('W and B must have the same size.')

    if M0 is None:
        M0 = np.arange(1, n + 1)
    elif len(M0) != n:
        raise ValueError('M0 must contain n elements.')

    _, _, Mb = np.unique(M0, return_index=True, return_inverse=True)
    M = Mb

    B = (B + B.T) / 2  # symmetrize modularity matrix
    Hnm = np.zeros((n, n))  # node-to-module degree
    for m in range(np.max(Mb)):  # loop over modules
        Hnm[:, m] = np.sum(B[:, Mb == m + 1], axis=1)

    Q0 = -np.inf
    Q = np.sum(B[np.arange(n), np.arange(n)]) # compute modularity
    first_iteration = True
    while Q - Q0 > 1e-10:
        flag = True  # flag for within-hierarchy search
        while flag:
            flag = False
            for u in np.random.permutation(n):  # loop over all nodes in random order
                ma = Mb[u] - 1  # current module of u
                dQ = Hnm[u, :] - Hnm[u, ma] + B[u, u]
                dQ[ma] = 0  # (line above) algorithm condition

                max_dQ, mb = np.max(dQ), np.argmax(dQ)  # maximal increase in modularity and corresponding module
                if max_dQ > 1e-10:  # if maximal increase is positive
                    flag = True
                    Mb[u] = mb + 1  # reassign module

                    Hnm[:, mb] = Hnm[:, mb] + B[:, u]  # change node-to-module strengths
                    Hnm[:, ma] = Hnm[:, ma] - B[:, u]
            
            _, _, Mb = np.unique(Mb, return_index=True, return_inverse=True)

        M0 = M
        if first_iteration:
            M = Mb
            first_iteration = False
        else:
            for u in range(n):  # loop through initial module assignments
                M[M0 == u + 1] = Mb[u]  # assign new modules

        n = np.max(Mb)  # new number of modules
        B1 = np.zeros((n, n))  # new weighted matrix
        for u in range(n):
            for v in range(u, n):
                bm = np.sum(B[Mb == u + 1, :][:, Mb == v + 1])  # pool weights of nodes in same module
                B1[u, v] = bm
                B1[v, u] = bm
        B = B1

        Mb = np.arange(1, n + 1)  # initial module assignments
        Hnm = B  # node-to-module strength

        Q0 = Q
        Q = np.trace(B)  # compute modularity

    return M, Q

