# Translated from navigation_wu.m

import numpy as np

def navigation_wu(L, D, max_hops=None):
    """
    Navigation of connectivity length matrix L guided by nodal distance D

    Parameters
    ----------
    L : numpy.ndarray
        Weighted/unweighted directed/undirected NxN SC matrix of connection lengths. 
        L(i,j) is the strength-to-length remapping of the connection weight between i and j. 
        L(i,j) = 0 denotes the lack of a connection between i and j.
    D : numpy.ndarray
        Symmetric NxN nodal distance matrix (e.g., Euclidean distance between node centroids).
    max_hops : int, optional
        Limits the maximum number of hops of navigation paths. If None, defaults to the size of L.

    Returns
    -------
    sr : float
        The success ratio (scalar) is the proportion of node pairs successfully reached by navigation.
    PL_bin : numpy.ndarray
        NxN matrix of binary navigation path length (i.e., number of hops in navigation paths). 
        Infinite values indicate failed navigation paths.
    PL_wei : numpy.ndarray
        NxN matrix of weighted navigation path length (i.e., sum of connection weights along navigation path). 
        Infinite values indicate failed navigation paths.
    PL_dis : numpy.ndarray
        NxN matrix of distance-based navigation path length (i.e., sum of connection distances along navigation path). 
        Infinite values indicate failed navigation paths.
    paths : list of lists
        NxN cell of nodes comprising navigation paths.

    """

    N = L.shape[0]
    if max_hops is None:
        max_hops = N

    paths = [([0] * N) for _ in range(N)]  # Initialize paths as a list of lists
    PL_bin = np.zeros((N, N))
    PL_wei = np.zeros((N, N))
    PL_dis = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                curr_node = i
                last_node = curr_node
                target = j
                paths[i][j] = [curr_node]

                pl_bin = 0
                pl_wei = 0
                pl_dis = 0

                while curr_node != target:
                    neighbors = np.where(L[curr_node, :] != 0)[0]

                    if len(neighbors) == 0:
                        pl_bin = np.inf
                        pl_wei = np.inf
                        pl_dis = np.inf
                        break

                    distances_to_target = D[target, neighbors]
                    min_index = np.argmin(distances_to_target)
                    next_node = neighbors[min_index]

                    if next_node == last_node or pl_bin > max_hops:
                        pl_bin = np.inf
                        pl_wei = np.inf
                        pl_dis = np.inf
                        break

                    paths[i][j].append(next_node)
                    pl_bin += 1
                    pl_wei += L[curr_node, next_node]
                    pl_dis += D[curr_node, next_node]

                    last_node = curr_node
                    curr_node = next_node

                PL_bin[i, j] = pl_bin
                PL_wei[i, j] = pl_wei
                PL_dis[i, j] = pl_dis

    np.fill_diagonal(PL_bin, np.inf)
    np.fill_diagonal(PL_wei, np.inf)
    np.fill_diagonal(PL_dis, np.inf)

    sr = 1 - (np.count_nonzero(PL_bin == np.inf) - N) / (N * N - N)

    return sr, PL_bin, PL_wei, PL_dis, paths

