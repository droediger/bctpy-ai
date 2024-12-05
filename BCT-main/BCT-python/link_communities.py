# Translated from link_communities.m

import numpy as np

def link_communities(W, type_clustering='single'):
    """Optimal overlapping community structure

    Args:
        W: Directed (weighted or binary) connection matrix.
        type_clustering: Type of hierarchical clustering ('single' or 'complete', default 'single').

    Returns:
        M: Nodal community-affiliation matrix (binary matrix of size CxN [communities x nodes]).

    Notes:
        The algorithm can be slow and memory intensive.
        Reference: Ahn, Bagrow and Lehmann (2010) Nature 466, 761–764.
    """

    # number of nodes
    n = W.shape[0]
    # set diagonal to 0
    np.fill_diagonal(W, 0)
    # normalize weights
    W = W / np.max(W)

    # get node similarity
    # mean weight on diagonal
    np.fill_diagonal(W, (np.sum(W) / np.sum(W != 0) + np.sum(W.T) / np.sum(W.T != 0)) / 2)
    # out-norm squared
    No = np.sum(W**2, axis=1)
    # in-norm squared
    Ni = np.sum(W**2, axis=0)

    # weighted in-Jaccard
    Jo = np.zeros((n, n))
    # weighted ou-Jaccard
    Ji = np.zeros((n, n))
    for b in range(n):
        for c in range(n):
            Do = np.dot(W[b, :], W[c, :])
            Jo[b, c] = Do / (No[b] + No[c] - Do)

            Di = np.dot(W[:, b].T, W[:, c])
            Ji[b, c] = Di / (Ni[b] + Ni[c] - Di)


    # get link similarity
    A, B = np.nonzero((W + W.T) * np.triu(np.ones((n, n)), 1))
    m = len(A)
    # link nodes
    Ln = np.zeros((m, 2), dtype=int)
    # link weights
    Lw = np.zeros(m)
    for i in range(m):
        Ln[i, :] = [A[i], B[i]]
        # link weight
        Lw[i] = (W[A[i], B[i]] + W[B[i], A[i]]) / 2

    # link similarity
    ES = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            if Ln[i, 0] == Ln[j, 0]:
                a = Ln[i, 0]
                b = Ln[i, 1]
                c = Ln[j, 1]
            elif Ln[i, 0] == Ln[j, 1]:
                a = Ln[i, 0]
                b = Ln[i, 1]
                c = Ln[j, 0]
            elif Ln[i, 1] == Ln[j, 0]:
                a = Ln[i, 1]
                b = Ln[i, 0]
                c = Ln[j, 1]
            elif Ln[i, 1] == Ln[j, 1]:
                a = Ln[i, 1]
                b = Ln[i, 0]
                c = Ln[j, 0]
            else:
                continue

            ES[i, j] = (W[a, b] * W[a, c] * Ji[b, c] + W[b, a] * W[c, a] * Jo[b, c]) / 2

    np.fill_diagonal(ES, 0)

    # perform hierarchical clustering
    # community affiliation matrix
    C = np.zeros((m, m), dtype=np.float32)
    Nc = np.copy(C)
    Mc = np.copy(C)
    Dc = np.copy(C)
    # initial community assignments
    U = np.arange(m)
    C[0, :] = U

    for i in range(m - 1):
        print(f'hierarchy {i+1:8d}')

        # compute densities
        for j in range(len(U)):
            idx = C[i, :] == U[j]
            links = np.sort(Lw[idx])
            nodes = np.sort(Ln[idx, :].reshape(-1))
            nodes = nodes[np.concatenate(([True], nodes[1:] != nodes[:-1]))]

            nc = len(nodes)
            mc = np.sum(links)
            min_mc = np.sum(links[:nc -1]) if nc > 1 else 0
            dc = (mc - min_mc) / (nc * (nc - 1) / 2 - min_mc) if nc > 1 else 0

            Nc[i, j] = nc
            Mc[i, j] = mc
            Dc[i, j] = dc

        # cluster
        C[i + 1, :] = C[i, :]
        u1, u2 = np.unravel_index(np.argmax(ES[np.ix_(U, U)]), ES[np.ix_(U, U)].shape)

        V = U[np.unique(np.sort(np.stack((u1, u2), axis=1), axis=1), axis=0)]

        for j in range(V.shape[0]):
            if type_clustering == 'single':
                x = np.max(ES[V[j, :], :], axis=0)
            elif type_clustering == 'complete':
                x = np.min(ES[V[j, :], :], axis=0)
            else:
                raise ValueError('Unknown clustering type.')

            ES[V[j, :], :] = np.vstack((x, x))
            ES[:, V[j, :]] = np.vstack((x, x)).T
            ES[V[j, 0], V[j, 0]] = 0
            ES[V[j, 1], V[j, 1]] = 0

            C[i + 1, C[i + 1, :] == V[j, 1]] = V[j, 0]
            V[V == V[j, 1]] = V[j, 0]

        U = np.unique(C[i + 1, :])
        if len(U) == 1:
            break

    Dc[np.isnan(Dc)] = 0
    i = np.argmax(np.sum(Dc * Mc, axis=1))

    U = np.unique(C[i, :])
    M = np.zeros((len(U), n))
    for j in range(len(U)):
        M[j, np.unique(Ln[C[i, :] == U[j], :])] = 1

    M = M[np.sum(M, axis=1) > 2, :]

    return M

