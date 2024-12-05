# Translated from modularity_dir.m

import numpy as np

def modularity_dir(A, gamma=1):
    """Optimal community structure and modularity

    Parameters
    ----------
    A : array_like
        Directed weighted/binary connection matrix
    gamma : float, optional
        Resolution parameter. 
        gamma > 1 detects smaller modules.
        0 <= gamma < 1 detects larger modules.
        gamma = 1 classic modularity (default).

    Returns
    -------
    Ci : ndarray
        Optimal community structure
    Q : float
        Maximized modularity

    Notes
    -----
    This algorithm is essentially deterministic. The only potential
    source of stochasticity occurs at the iterative finetuning step, in
    the presence of non-unique optimal swaps. However, the present
    implementation always makes the first available optimal swap and
    is therefore deterministic.

    References
    ----------
    Leicht and Newman (2008) Phys Rev Lett 100:118703.
    Reichardt and Bornholdt (2006) Phys Rev E 74:016110.
    """

    N = len(A)  # number of vertices
    Ki = np.sum(A, axis=1)  # in-degree
    Ko = np.sum(A, axis=0)  # out-degree
    m = np.sum(Ki)  # number of edges
    b = A - gamma * (np.outer(Ko, Ki)) / m
    B = b + b.T  # directed modularity matrix
    Ci = np.ones(N)  # community indices
    cn = 1  # number of communities
    U = np.array([1, 0])  # array of unexamined communities

    ind = np.arange(N)
    Bg = B
    Ng = N

    while U[0]:  # examine community U[0]
        V, D = np.linalg.eig(Bg)
        i1 = np.argmax(np.real(np.diag(D)))  # maximal positive (real part of) eigenvalue of Bg
        v1 = V[:, i1]  # corresponding eigenvector

        S = np.ones(Ng)
        S[v1 < 0] = -1
        q = np.dot(S.T, np.dot(Bg, S))  # contribution to modularity

        if q > 1e-10:  # contribution positive: U[0] is divisible
            qmax = q  # maximal contribution to modularity
            np.fill_diagonal(Bg, 0)  # Bg is modified, to enable fine-tuning
            indg = np.ones(Ng)  # array of unmoved indices
            Sit = S
            while np.any(indg):  # iterative fine-tuning
                Qit = qmax - 4 * Sit * (Bg @ Sit)
                qmax, imax = np.max(Qit * indg), np.argmax(Qit * indg)
                Sit[imax] *= -1
                indg[imax] = np.nan
                if qmax > q:
                    q = qmax
                    S = Sit

            if np.isclose(np.abs(np.sum(S)), Ng):  # unsuccessful splitting of U[0]
                U = U[1:]
            else:
                cn += 1
                Ci[ind[S == 1]] = U[0]  # split old U[0] into new U[0] and into cn
                Ci[ind[S == -1]] = cn
                U = np.append(cn, U)

        else:  # contribution nonpositive: U[0] is indivisible
            U = U[1:]

        ind = np.where(Ci == U[0])[0]  # indices of unexamined community U[0]
        bg = B[np.ix_(ind, ind)]
        Bg = bg - np.diag(np.sum(bg, axis=1))  # modularity matrix for U[0]
        Ng = len(ind)  # number of vertices in U[0]

    s = np.tile(Ci, (1, N))  # compute modularity
    Q = (1 - np.equal(*np.indices(s.shape))) * B / (2 * m)
    Q = np.sum(Q)

    return Ci, Q

