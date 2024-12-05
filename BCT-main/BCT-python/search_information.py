# Translated from search_information.m

import numpy as np

def search_information(W, L, has_memory=False):
    """
    Search information

    Computes the amount of information (measured in bits) that a random
    walker needs to follow the shortest path between a given pair of nodes.

    Args:
        W (numpy.ndarray): Weighted/unweighted directed/undirected connection weight matrix.
        L (numpy.ndarray): Weighted/unweighted directed/undirected connection length matrix.
        has_memory (bool, optional): This flag defines whether or not the random walker "remembers" its previous step. Defaults to False.

    Returns:
        numpy.ndarray: Pair-wise search information (matrix). Note that SI(i,j) may be different from SI(j,i), hence, SI is not a symmetric matrix even when adj is symmetric.

    References: Rosvall et al. (2005) Phys Rev Lett 94, 028701
                Goñi et al (2014) PNAS doi: 10.1073/pnas.131552911
    """

    N = W.shape[0]

    # Determine if W is symmetric
    flag_triu = np.allclose(W, W.T)

    # Compute transition matrix
    T = np.linalg.solve(np.diag(np.sum(W, axis=1)), W)

    # Assume distance_wei_floyd is defined elsewhere and returns hops and Pmat
    hops, Pmat = distance_wei_floyd(L) # Compute shortest paths based on L

    SI = np.zeros((N, N))
    np.fill_diagonal(SI, np.nan)

    for i in range(N):
        for j in range(N):
            if (j > i and flag_triu) or (not flag_triu and i != j):
                path = retrieve_shortest_path(i, j, hops, Pmat)
                lp = len(path)
                if lp > 0:
                    pr_step_ff = np.empty(lp - 1)
                    pr_step_ff[:] = np.nan
                    pr_step_bk = np.empty(lp - 1)
                    pr_step_bk[:] = np.nan

                    if has_memory:
                        pr_step_ff[0] = T[path[0], path[1]]
                        pr_step_bk[lp - 2] = T[path[-1], path[-2]]
                        for z in range(1, lp - 1):
                            pr_step_ff[z] = T[path[z], path[z + 1]] / (1 - T[path[z - 1], path[z]])
                            pr_step_bk[lp - 2 - z] = T[path[lp - z], path[lp - 1 - z]] / (1 - T[path[lp - z + 1], path[lp - z]])
                    else:
                        for z in range(lp - 1):
                            pr_step_ff[z] = T[path[z], path[z + 1]]
                            pr_step_bk[z] = T[path[z + 1], path[z]]

                    prob_sp_ff = np.prod(pr_step_ff)
                    prob_sp_bk = np.prod(pr_step_bk)
                    SI[i, j] = -np.log2(prob_sp_ff)
                    SI[j, i] = -np.log2(prob_sp_bk)
                else:
                    SI[i, j] = np.inf
                    SI[j, i] = np.inf
    return SI


# Placeholder functions - replace with actual implementations
def distance_wei_floyd(L, extra_arg=None):
    # Replace with your actual implementation of the Floyd-Warshall algorithm
    # This should compute shortest paths and return the hop count and path matrix
    N = L.shape[0]
    hops = np.zeros((N,N), dtype=int)
    Pmat = np.zeros((N,N), dtype=int)
    #Example, replace with actual calculation
    for i in range(N):
        for j in range(N):
            if i!=j:
                hops[i,j] = 1
    return hops, Pmat


def retrieve_shortest_path(i, j, hops, Pmat):
    # Replace with your actual implementation to retrieve shortest path
    #This should take the start and end node and return a list of nodes
    if hops[i,j] == 0:
      return []
    else:
      return list(range(i,j+1))


