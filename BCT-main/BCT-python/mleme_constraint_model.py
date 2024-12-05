# Translated from mleme_constraint_model.m

import numpy as np
from scipy.optimize import fsolve
from numpy import mean

def mleme_constraint_model(samp, W, M=None, Lo=None, Li=None, Lm=None, opts=None):
    """Unbiased sampling of networks with soft constraints.

    This function returns an ensemble of unbiasedly sampled networks with
    weighted node-strength and module-weight constraints. These constraints
    are soft in that they are satisfied on average for the full network
    ensemble but not, in general, for each individual network.

    Args:
        samp: Number of networks to sample.
        W: (n x n) square directed and weighted connectivity matrix. All weights must be nonnegative integers.
        M: (length n) module affiliation vector.  Defaults to all nodes in one module if not provided.
        Lo: (length n) out-strength constraint logical vector. Defaults to no constraints if not provided.
        Li: (length n) in-strength constraint logical vector. Defaults to no constraints if not provided.
        Lm: (m x m) module-weight constraint logical matrix. Defaults to no constraints if not provided.
        opts: optional argument: pass optimization and display options with optimset.  Defaults to a reasonable set of options if not provided.


    Returns:
        W0: an ensemble of sampled networks with constraints.
        E0: expected weights matrix.
        P0: probability matrix.
        Delt0: algorithm convergence error.
    """
    n = len(W)  # number of nodes

    if M is None or M.size == 0:
        if Lm is not None and np.any(Lm):
            raise ValueError('Need module affiliation vector for module constraints')
        else:
            M = np.zeros(n, dtype=int)
    m = np.max(M)  # number of modules

    if not np.allclose(W, np.round(W)) or np.min(W) < 0:
        raise ValueError('W must only contain nonnegative integers.')
    if not np.allclose(M, np.round(M)) or np.min(M) < 0:
        raise ValueError('M must only contain nonnegative integers.')

    # process node constraints
    if Lo is None or Lo.size == 0 or Lo == 0:
        Lo = np.zeros(n, dtype=bool)
    elif Lo == 1:
        Lo = np.ones(n, dtype=bool)
    if Li is None:
        Li = Lo
    elif Li.size == 0 or Li == 0:
        Li = np.zeros(n, dtype=bool)
    elif Li == 1:
        Li = np.ones(n, dtype=bool)


    # process module constraints
    if Lm is None or Lm.size == 0 or Lm == 0:
        Lm = np.zeros((m, m), dtype=bool)
    elif Lm == 2:
        Lm = np.ones((m, m), dtype=bool)
    elif Lm == 1:
        Lm = np.eye(m, dtype=bool)
    if np.any(~M):
        m = m + 1
        M = np.where(M==0, m,M)
        Lm = np.pad(Lm, ((0,1),(0,1)), mode='constant')
        Lm[-1,-1] = 0  # add a new row and column for nodes without modules

    Lo = Lo.astype(bool)
    Li = Li.astype(bool)
    Lm = Lm.astype(bool)
    ao = len(Lo)
    ai = len(Li)
    am = len(Lm.ravel())
    uo = np.sum(Lo)
    ui = np.sum(Li)
    um = np.sum(Lm)
    Mij = M[:, np.newaxis] + (M.T - 1) * m

    def f_ex(V):
        return system_equations(V, Mij, Lo, Li, Lm, ao, ai, am, uo, ui, um)

    def f_cx(W):
        return system_constraints(W, M, Lo, Li, Lm, uo, ui, um)

    C = f_cx(W)
    c = 1 + uo + ui + um
    V = (mean(W.ravel())/(1+mean(W.ravel())))*np.ones(c)


    assert c == len(C)
    assert c == len(V)

    if opts is None or opts.size ==0:
        opts = {'maxfev': int(1e6 * c), 'maxiter': int(1e6), 'disp': True}

    V0 = fsolve(lambda V: C - f_cx(f_ex(V)), V, **opts)

    E0, P0 = f_ex(V0)
    Delt0 = C - f_cx(f_ex(V0))

    W0 = sample_networks(P0, samp)

    return W0, E0, P0, Delt0


def sample_networks(P0, samp):
    n = len(P0)
    CellW0 = []
    for i in range(samp):
        W0 = np.zeros((n, n))
        L0 = np.ones((n, n), dtype=bool) - np.eye(n)
        while np.any(L0):
            l0 = np.sum(L0)
            idx = np.nonzero(L0)
            L0[idx] = P0[idx] > np.random.rand(l0)
            W0[idx] = W0[idx] + 1

        CellW0.append(W0)
    return CellW0



def system_equations(V, Mij, Lo, Li, Lm, ao, ai, am, uo, ui, um):
    X = np.ones(ao)
    Y = np.ones(ai)
    Z = np.ones(am)

    if uo:
        offset = 1
        X[Lo] = V[offset + (np.arange(uo))]
    if ui:
        offset = 1 + uo
        Y[Li] = V[offset + (np.arange(ui))]
    if um:
        offset = 1 + uo + ui
        Z[Lm] = V[offset + (np.arange(um))]
    P = V[0] * (X[:, np.newaxis] * Y) * Z[Mij.ravel()].reshape(Mij.shape)
    P[P > 1] = 1 - np.finfo(float).eps

    W = P / (1 - P)
    np.fill_diagonal(W, 0)

    return W, P


def system_constraints(W, M, Lo, Li, Lm, uo, ui, um):
    if uo:
        So = np.sum(W[Lo, :], axis=1)
    else:
        So = np.array([])
    if ui:
        Si = np.sum(W[:, Li], axis=0)
    else:
        Si = np.array([])
    if um:
        Wm = block_density(W, M, Lm)
    else:
        Wm = np.array([])
    C = np.concatenate((np.array([np.sum(W.ravel())]), So, Si, Wm))
    return C


def block_density(W, M, Lwm):
    m = np.max(M)
    Wm = np.zeros(m * m)
    for u in range(m):
        for v in range(m):
            Wm[u + v * m] = np.sum(W[M == u+1, M == v+1])
    return Wm[Lwm]


