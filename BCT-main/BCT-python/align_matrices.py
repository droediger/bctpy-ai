# Translated from align_matrices.m

import numpy as np

def align_matrices(M1, M2, dfun, flag):
    """Alignment of two matrices

    Args:
        M1 (numpy.ndarray): First connection matrix (square).
        M2 (numpy.ndarray): Second connection matrix (square).
        dfun (str): Distance metric to use for matching:
                     'absdff' = absolute difference
                     'sqrdff' = squared difference
                     'cosang' = cosine of vector angle
        flag (int): Flag to control output display (1 for display, 0 for no display).

    Returns:
        tuple: A tuple containing:
            - Mreordered (numpy.ndarray): Reordered connection matrix M2.
            - Mindices (numpy.ndarray): Reordered indices.
            - cost (float): Distance between M1 and Mreordered.

    Connection matrices can be weighted or binary, directed or undirected.
    They must have the same number of nodes. M1 can be entered in any node ordering.

    Note that in general, the outcome will depend on the initial condition
    (the setting of the random number seed). Also, there is no good way to
    determine optimal annealing parameters in advance - these parameters
    will need to be adjusted "by hand" (particularly H, Texp, T0, and Hbrk).
    For large and/or dense matrices, it is highly recommended to perform
    exploratory runs varying the settings of 'H' and 'Texp' and then select
    the best values.

    Based on extensive testing, it appears that T0 and Hbrk can remain
    unchanged in most cases. Texp may be varied from 1-1/H to 1-10/H, for
    example. H is the most important parameter - set to larger values as
    the problem size increases. Good solutions can be obtained for
    matrices up to about 100 nodes. It is advisable to run this function
    multiple times and select the solution(s) with the lowest 'cost'.

    If the two matrices are related it may be very helpful to pre-align them
    by reordering along their largest eigenvectors:
        [v,~] = eig(M1); v1 = abs(v(:,end)); [a1,b1] = sort(v1);
        [v,~] = eig(M2); v2 = abs(v(:,end)); [a2,b2] = sort(v2);
        [a,b,c] = overlapMAT2(M1(b1,b1),M2(b2,b2),'dfun',1);

    Setting 'Texp' to zero cancels annealing and uses a greedy algorithm
    instead.
    """
    N = M1.shape[0]

    # Define maxcost (greatest possible difference)
    if dfun == 'absdff':
        maxcost = np.sum(np.abs(np.sort(M1.flatten()) - np.sort(M2.flatten())[::-1]))
    elif dfun == 'sqrdff':
        maxcost = np.sum((np.sort(M1.flatten()) - np.sort(M2.flatten())[::-1])**2)
    elif dfun == 'cosang':
        maxcost = np.pi/2
    else:
        raise ValueError("Invalid dfun specified.")


    # Initialize lowcost
    if dfun == 'absdff':
        lowcost = np.sum(np.abs(M1 - M2)) / maxcost
    elif dfun == 'sqrdff':
        lowcost = np.sum((M1 - M2)**2) / maxcost
    elif dfun == 'cosang':
        lowcost = np.arccos(np.dot(M1.flatten(), M2.flatten()) / np.sqrt(np.dot(M1.flatten(), M1.flatten()) * np.dot(M2.flatten(), M2.flatten()))) / maxcost
    else:
        raise ValueError("Invalid dfun specified.")


    # Initialize
    mincost = lowcost
    anew = np.arange(N)
    amin = np.arange(N)
    h = 0
    hcnt = 0

    # Set annealing parameters
    H = 1e6
    Texp = 1 - 1/H
    T0 = 1e-3
    Hbrk = H/10
    #Texp = 0  # Uncomment to use greedy algorithm

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
        r = np.random.randint(1, N + 1, size=2)
        atmp[r[0]-1], atmp[r[1]-1] = atmp[r[1]-1], atmp[r[0]-1]

        if dfun == 'absdff':
            costnew = np.sum(np.abs(M1 - M2[np.ix_(atmp, atmp)])) / maxcost
        elif dfun == 'sqrdff':
            costnew = np.sum((M1 - M2[np.ix_(atmp, atmp)])**2) / maxcost
        elif dfun == 'cosang':
            M2atmp = M2[np.ix_(atmp, atmp)]
            costnew = np.arccos(np.dot(M1.flatten(), M2atmp.flatten()) / np.sqrt(np.dot(M1.flatten(), M1.flatten()) * np.dot(M2atmp.flatten(), M2atmp.flatten()))) / maxcost
        else:
            raise ValueError("Invalid dfun specified.")

        # Annealing step
        if (costnew < lowcost) or (np.random.rand() < np.exp(-(costnew - lowcost) / T)):
            anew = atmp.copy()
            lowcost = costnew
            # Is this the absolute best?
            if lowcost < mincost:
                amin = anew.copy()
                mincost = lowcost
                if flag == 1:
                    print(f'step {h} ... current lowest cost = {mincost}')
                hcnt = 0
            # If the cost is 0 we're done
            if mincost == 0:
                break

    print(f'step {h} ... final lowest cost = {mincost}')

    # Prepare output
    Mreordered = M2[np.ix_(amin, amin)]
    Mindices = amin + 1 # Adjust to 1-based indexing
    cost = mincost
    return Mreordered, Mindices, cost


