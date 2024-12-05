# Translated from modularity_und.m

import numpy as np

def modularity_und(A, gamma=1):
    """
    Optimal community structure and modularity.

    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes in a way that maximizes the number of
    within-group edges, and minimizes the number of between-group edges.
    The modularity is a statistic that quantifies the degree to which the
    network may be subdivided into such clearly delineated groups.

    Args:
        A (numpy.ndarray): Undirected weighted/binary connection matrix.
        gamma (float, optional): Resolution parameter. 
                                  gamma > 1 detects smaller modules;
                                  0 <= gamma < 1 detects larger modules;
                                  gamma = 1 is classic modularity (default).

    Returns:
        tuple: A tuple containing:
            Ci (numpy.ndarray): Optimal community structure.
            Q (float): Maximized modularity.
    """

    N = len(A)  # number of vertices
    K = np.sum(A, axis=1)  # degree
    m = np.sum(K)  # number of edges (each undirected edge is counted twice)
    B = A - gamma * (np.outer(K, K) / m)  # modularity matrix
    Ci = np.ones(N, dtype=int)  # community indices
    cn = 1  # number of communities
    U = np.array([1, 0])  # array of unexamined communities

    ind = np.arange(N)
    Bg = B
    Ng = N

    while U[0]:  # examine community U[0]
        V, D = np.linalg.eig(Bg)
        i1 = np.argmax(np.real(np.diag(D)))  # maximal positive (real part of) eigenvalue of Bg
        v1 = V[:, i1]  # corresponding eigenvector

        S = np.ones(Ng, dtype=int)
        S[v1 < 0] = -1
        q = np.dot(S.T, np.dot(Bg, S))  # contribution to modularity

        if q > 1e-10:  # contribution positive: U[0] is divisible
            qmax = q  # maximal contribution to modularity
            np.fill_diagonal(Bg, 0)  # Bg is modified, to enable fine-tuning
            indg = np.ones(Ng, dtype=int)  # array of unmoved indices
            Sit = S
            while np.any(indg):  # iterative fine-tuning
                Qit = qmax - 4 * Sit * (Bg @ Sit)
                qmax, imax = np.max(Qit * indg), np.argmax(Qit * indg)
                Sit[imax] *= -1
                indg[imax] = np.nan
                if qmax > q:
                    q = qmax
                    S = Sit

        if np.abs(np.sum(S)) == Ng:  # unsuccessful splitting of U[0]
            U = U[1:]
        else:
            cn += 1
            Ci[ind[S == 1]] = U[0]  # split old U[0] into new U[0] and into cn
            Ci[ind[S == -1]] = cn
            U = np.concatenate(([cn], U))

        ind = np.where(Ci == U[0])[0]  # indices of unexamined community U[0]
        bg = B[np.ix_(ind, ind)]
        Bg = bg - np.diag(np.sum(bg, axis=1))  # modularity matrix for U[0]
        Ng = len(ind)  # number of vertices in U[0]

    s = np.tile(Ci, (1, N))  # compute modularity
    Q = (s != s.T).astype(int) * B / m
    Q = np.sum(Q)

    return Ci, Q

