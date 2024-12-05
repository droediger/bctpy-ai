# Translated from reorder_mod.m

import numpy as np

def reorder_mod(W, M):
    # REORDER_MOD         Reorder connectivity matrix by modular structure
    #
    #   On = reorder_mod(W,M);
    #   [On Wr] = reorder_mod(W,M);
    #
    #   This function reorders the connectivity matrix by modular structure and
    #   may consequently be useful in visualization of modular structure.
    #
    #   Inputs:
    #       W,      connectivity matrix (binary/weighted undirected/directed)
    #       M,      module affiliation vector
    #
    #   Outputs:
    #       On,     new node order
    #       Wr,     reordered connectivity matrix
    #
    #
    #   Used in: Rubinov and Sporns (2011) NeuroImage; Zingg et al. (2014) Cell.
    #
    #
    #   2011, Mika Rubinov, UNSW/U Cambridge

    #   Modification History:
    #   Mar 2011: Original
    #   Jan 2015: Improved behavior for directed networks

    W = W + np.finfo(float).eps  #add a small value to avoid division by zero

    u, dum, M = np.unique(M, return_index=True, return_inverse=True) #make consecutive
    n = len(M)  #number of nodes
    m = len(u)  #number of modules

    Nm = np.zeros(m)  #number of nodes in modules
    Knm_o = np.zeros((n, m))  #node-to-module out-degree
    Knm_i = np.zeros((n, m))  #node-to-module in-degree
    for i in range(m):
        Nm[i] = np.sum(M == i + 1)
        Knm_o[:, i] = np.sum(W[:, M == i + 1], axis=1)
        Knm_i[:, i] = np.sum(W[M == i + 1, :], axis=0)
    Knm = (Knm_o + Knm_i) / 2

    Wm = np.zeros((m, m))
    for u in range(m):
        for v in range(m):
            Wm[u, v] = np.sum(W[M == u + 1, M == v + 1])
    Bm = (Wm + Wm.T) / (2 * np.outer(Nm, Nm))

    #1. Arrange densely connected modules together
    I, J, bv = np.where(np.tril(Bm, -1))  #symmetrized intermodular connectivity values
    ord = np.argsort(bv)[::-1]  #sort by greatest relative connectivity
    I = I[ord]
    J = J[ord]
    Om = np.array([I[0], J[0]])  #new module order

    Si = np.ones(len(I), dtype=bool)
    Sj = np.ones(len(J), dtype=bool)
    Si[np.isin(I, Om)] = False
    Sj[np.isin(J, Om)] = False


    while len(Om) < m:  #while not all modules ordered
        for u in range(len(I)):
            if Si[u] and np.any(J[u] == Om[[0, -1]]):
                old = J[u]
                new = I[u]
                break
            elif Sj[u] and np.any(I[u] == Om[[0, -1]]):
                old = I[u]
                new = J[u]
                break
        if old == Om[0]:
            Om = np.concatenate(([new], Om))
        elif old == Om[-1]:
            Om = np.concatenate((Om, [new]))
        Si[I == new] = False
        Sj[J == new] = False

    #2. Reorder nodes within modules
    On = np.zeros(n, dtype=np.uint64)  #node order array
    for i in range(m):
        u = Om[i]
        ind = np.where(M == u + 1)[0]  #indices

        mod_imp = np.vstack((Om, np.sign(np.arange(m) - i), np.abs(np.arange(m) - i), Bm[u, Om])).T
        mod_imp = mod_imp[mod_imp[:,2].argsort()]
        mod_imp = np.prod(mod_imp[:,1:], axis =1)

        ord = np.argsort(Knm[ind, :], axis=1)[:,mod_imp.argsort()[::-1][0]]
        On[ind[ord]] = 1000000 * i + np.arange(1, Nm[i] +1) #assign node order (assumes <1e6 nodes in a module)

    ord = np.argsort(On)
    On = On[ord]  #reorder nodes
    Wr = W[ord, :][:, ord]  #reorder matrix

    return On, Wr

