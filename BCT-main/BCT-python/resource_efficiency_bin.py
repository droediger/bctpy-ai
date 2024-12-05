# Translated from resource_efficiency_bin.m

import numpy as np

def resource_efficiency_bin(adj, lambda_val, SPL=None, M=None):
    """
    Resource efficiency and shortest-path probability

    Parameters:
        adj (numpy.ndarray): Unweighted, undirected adjacency matrix.
        lambda_val (float): Probability (0 < lambda < 1). Set to np.nan if computation of Eres is not desired.
        SPL (numpy.ndarray, optional): Shortest-path length matrix. Defaults to None.
        M (numpy.ndarray, optional): Transition probability matrix. Defaults to None.

    Returns:
        tuple: A tuple containing:
            Eres (numpy.ndarray): Resource efficiency matrix.
            prob_SPL (numpy.ndarray): Shortest-path probability matrix.
    """

    N = adj.shape[0]
    EYE = np.eye(N, dtype=bool)

    flagResources = not np.isnan(lambda_val)

    if flagResources:
        if lambda_val <= 0 or lambda_val >= 1:
            raise ValueError('lambda_val must be a non-zero probability')
        z = np.zeros((N, N))

    if SPL is None:
        SPL = distance_wei_floyd(adj)  # Assume distance_wei_floyd is defined elsewhere

    if M is None:
        M = np.linalg.solve(np.diag(np.sum(adj, axis=1)), adj)

    Lvalues = np.unique(SPL)
    Lvalues = Lvalues[Lvalues != 0]

    prob_SPL = np.zeros((N, N))  # Initialize shortest-path probability matrix

    for SPLvalue in Lvalues:  # Iterate through each possible SPL value
        rows, cols = np.where(SPL == SPLvalue)
        hvector = np.unique(cols)

        entries = (SPL == SPLvalue)

        if flagResources:  # Compute Eres
            prob_aux, z_aux = prob_first_particle_arrival(M, SPLvalue, hvector, lambda_val)
        else:  # Do not compute Eres
            prob_aux = prob_first_particle_arrival(M, SPLvalue, hvector, [])

        prob_aux[~entries] = 0
        prob_SPL += prob_aux

        if flagResources:
            z_aux[~entries] = 0
            z += z_aux

    prob_SPL[EYE] = 0

    if flagResources:
        z[prob_SPL == 1] = 1
        Eres = 1. / z
        Eres[EYE] = 0
    else:
        Eres = np.nan

    return Eres, prob_SPL


def prob_first_particle_arrival(M, L, hvector, lambda_val):
    """
    Probability of first particle arrival.

    Parameters:
        M (numpy.ndarray): Transition probability matrix.
        L (int): Shortest-path length.
        hvector (numpy.ndarray): Destination nodes.
        lambda_val (float or None): Probability (0 < lambda < 1). Set to None if computation of resources is not desired.

    Returns:
        tuple: A tuple containing:
            prob (numpy.ndarray): Probability matrix.
            resources (numpy.ndarray): Resource matrix (only if lambda_val is provided).
    """

    N = M.shape[0]
    prob = np.zeros((N, N))

    if lambda_val is None:
        hvector = np.arange(N)

    flagResources = lambda_val is not None

    if flagResources:
        if lambda_val <= 0 or lambda_val >= 1:
            raise ValueError('lambda_val must be a non-zero probability')
        resources = np.zeros((N, N))


    for h in hvector:  # Iterate through destination nodes
        B_h = np.copy(M)
        B_h[h, :] = 0
        B_h[h, h] = 1  # Make h an absorbing state

        B_h_L = np.linalg.matrix_power(B_h, L)

        term = 1 - B_h_L[:, h]

        prob[:, h] = 1 - term

        if flagResources:
            resources[:, h] = np.log(1 - lambda_val) / np.log(term)

    if flagResources:
        return prob, resources
    else:
        return prob


# Placeholder for distance_wei_floyd, replace with actual implementation if available.
def distance_wei_floyd(adj):
    # Replace with the actual implementation of the Floyd-Warshall algorithm
    # This is a placeholder, and the implementation might vary.
    N = adj.shape[0]
    dist = np.full((N, N), np.inf)
    np.fill_diagonal(dist, 0)
    for i in range(N):
      for j in range(N):
        if adj[i,j] ==1:
          dist[i,j] = 1
    for k in range(N):
        for i in range(N):
            for j in range(N):
                dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
    return dist

