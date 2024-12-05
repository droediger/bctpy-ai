# Translated from reorder_matrix.m

import numpy as np

def reorder_matrix(M1, cost_type, flag):
    """
    Matrix reordering for visualization.

    Rearranges the nodes in matrix M1 such that the matrix elements are 
    squeezed along the main diagonal using simulated annealing.

    Args:
        M1 (numpy.ndarray): Connection matrix (weighted or binary, directed or undirected).
        cost_type (str): 'line' or 'circ', for shape of lattice cost (linear or ring lattice).
        flag (int): Flag to control printing of intermediate results (1 for printing, 0 for no printing).

    Returns:
        tuple: A tuple containing:
            Mreordered (numpy.ndarray): Reordered connection matrix.
            Mindices (numpy.ndarray): Reordered indices.
            cost (float): Distance between M1 and Mreordered.
    """
    N = M1.shape[0]

    # Generate cost function
    if cost_type == 'line':
        profil = np.flip(np.exp(-0.5 * ((np.arange(1, N + 1) - 0.5 * N) / (0.5 * N))**2))
    elif cost_type == 'circ':
        profil = np.flip(np.exp(-0.5 * ((np.arange(1, N + 1) - 0.5 * N - 0.25*N) / (0.25 * N))**2))
    else:
        raise ValueError("Invalid cost_type. Choose 'line' or 'circ'.")

    COST = np.outer(profil, profil) * (1 - np.eye(N))
    COST = COST / np.sum(COST)

    # Establish maxcost, lowcost, mincost
    maxcost = np.sum(np.sort(COST.flatten()) * np.sort(M1.flatten()))
    lowcost = np.sum(M1 * COST) / maxcost
    mincost = lowcost

    # Initialize
    anew = np.arange(1, N + 1)
    amin = np.arange(1, N + 1)
    h = 0
    hcnt = 0

    # Set annealing parameters
    H = 10000
    Texp = 1 - 10 / H
    T0 = 1e-3
    Hbrk = H / 10
    # Texp = 0  #Uncomment for greedy algorithm


    while h < H:
        h += 1
        hcnt += 1
        # Terminate if no new mincost has been found for some time
        if hcnt > Hbrk:
            break

        # Current temperature
        T = T0 * Texp**h

        # Choose two positions at random and flip them
        atmp = anew.copy()
        r = np.random.randint(1, N + 1, 2)
        atmp[r[0] - 1], atmp[r[1] - 1] = atmp[r[1] - 1], atmp[r[0] - 1]
        costnew = np.sum(M1[atmp - 1, :][:, atmp - 1] * COST) / maxcost

        # Annealing
        if (costnew < lowcost) or (np.random.rand() < np.exp(-(costnew - lowcost) / T)):
            anew = atmp.copy()
            lowcost = costnew
            # Is this a new absolute best?
            if lowcost < mincost:
                amin = anew.copy()
                mincost = lowcost
                if flag == 1:
                    print(f'step {h} ... current lowest cost = {mincost}')
                hcnt = 0

    print(f'step {h} ... final lowest cost = {mincost}')

    # Prepare output
    Mreordered = M1[amin - 1, :][:, amin - 1]
    Mindices = amin
    cost = mincost
    return Mreordered, Mindices, cost

