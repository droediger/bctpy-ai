# Translated from randomizer_bin_und.m

import numpy as np

def randomizer_bin_und(R, alpha):
    """Random graph with preserved in/out degree distribution.

    Args:
        R: Binary undirected connection matrix.
        alpha: Fraction of edges to rewire.

    Returns:
        Randomized network.

    References: Maslov and Sneppen (2002) Science 296:910
    """

    # Make binary
    R = np.ceil(R)

    # Ensure that matrix is binary
    if (np.max(R) != 1) or (np.min(R) != 0):
        raise ValueError('Matrix should be binary')

    # Ensure that matrix is undirected
    if not np.array_equal(R, R.T):
        raise ValueError('Matrix should be undirected')

    # Find how many edges are possible in the network
    a, b = R.shape
    numpossibleedges = ((a * a) - a) // 2

    # Excise the diagonal and replace it with 9999
    savediag = R * np.eye(a)
    R = R * (1 - np.eye(a))
    R = R + (np.eye(a) * 9999)

    # If there are more edges than non-edges we invert the matrix to reduce
    # computation time, then revert at the end of the script
    inverted = 0
    i, j = np.nonzero(np.triu(R, 1))
    K = len(i)
    if K > (numpossibleedges / 2):
        inverted = 1
        R = 1 - R.astype(float)
        R = R * (1 - np.eye(a))
        R = R + (np.eye(a) * 9999)

    # Find edges
    i, j = np.nonzero(np.triu(R, 1))

    # Exclude fully connected nodes. Will replace later
    fullnode = np.where((np.sum(np.triu(R, 1), axis=1) + np.sum(np.triu(R, 1), axis=0)) == (a - 1))[0]
    if len(fullnode) > 0:
        R[fullnode, :] = 0
        R[:, fullnode] = 0
        R = R * (1 - np.eye(a))
        R = R + (np.eye(a) * 9999)

    # Find the edges
    i, j = np.nonzero(np.triu(R, 1))
    K = len(i)

    if (len(K) == 0 or K == numpossibleedges or K == numpossibleedges - 1):
        print('No possible randomization.')
    else:
        for iter in range(K):  # For every edge
            if np.random.rand() <= alpha:  # Rewire ~alpha% of edges

                # This is the chosen edge
                a = i[iter]
                b = j[iter]

                # For selected edge, see where each end can connect to
                alliholes = np.where(R[:, i[iter]] == 0)[0]
                alljholes = np.where(R[:, j[iter]] == 0)[0]

                # We can only use edges with connection to neither node
                iintersect = np.intersect1d(alliholes, alljholes)

                # Find which of these nodes are connected
                ii, jj = np.nonzero(R[iintersect, :][:, iintersect] == 1)

                # If there an edge to switch
                if len(ii) > 0:

                    # Choose one randomly
                    nummates = len(ii)
                    mate = np.ceil(np.random.rand() * nummates).astype(int)

                    # Randomly orient the second edge
                    if np.random.rand() < 0.5:
                        c = iintersect[ii[mate -1]]
                        d = iintersect[jj[mate -1]]
                    else:
                        d = iintersect[ii[mate -1]]
                        c = iintersect[jj[mate -1]]

                    # Make the changes in the matrix
                    R[a, b] = 0
                    R[c, d] = 0
                    R[b, a] = 0
                    R[d, c] = 0
                    R[a, c] = 1
                    R[b, d] = 1
                    R[c, a] = 1
                    R[d, b] = 1


        # Restore full columns
    if len(fullnode) > 0:
        R[fullnode, :] = 1
        R[:, fullnode] = 1

    # If we did non-edges switch it back to edges
    if inverted == 1:
        R = 1 - R.astype(float)

    # Clear and restore the diagonal
    R = R * (1 - np.eye(a))
    R = R + savediag

    return R

