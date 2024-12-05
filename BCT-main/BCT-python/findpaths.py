# Translated from findpaths.m

import numpy as np

def findpaths(CIJ, sources, qmax, savepths):
    """
    Network paths

    Finds all paths starting from a set of source nodes, up to a specified length.  Warning: very memory-intensive.

    Args:
        CIJ (numpy.ndarray): Binary (directed/undirected) connection matrix.
        qmax (int): Maximal path length.
        sources (numpy.ndarray): Source units from which paths are grown.
        savepths (int): Set to 1 if all paths are to be collected in 'allpths'.

    Returns:
        tuple: A tuple containing:
            Pq (numpy.ndarray): 3D matrix, where Pq[i, j, q] = number of paths from 'i' to 'j' of length 'q'.
            tpath (int): Total number of paths found (lengths 1 to 'qmax').
            plq (numpy.ndarray): Path length distribution as a function of 'q'.
            qstop (int): Path length at which 'findpaths' is stopped.
            allpths (numpy.ndarray): A matrix containing all paths up to 'qmax' (if savepths == 1).
            util (numpy.ndarray): Node use index.

    Notes:
        Pq[:, :, N] can only carry entries on the diagonal, as all "legal" paths of length N-1 must terminate. Cycles of length N are possible, with all vertices visited exactly once (except for source and target). 'qmax = N' can wreak havoc (due to memory problems).
        Weights are discarded.
        This algorithm is rather inefficient.
    """

    # Ensure CIJ is binary
    CIJ = (CIJ != 0).astype(int)

    # Initialize some variables
    N = CIJ.shape[0]
    K = np.sum(CIJ)
    pths = np.array([]).reshape(2,0) # Initialize as an empty 2D array
    Pq = np.zeros((N, N, qmax))
    util = np.zeros((N, qmax))

    # Paths of length 1
    q = 1
    for j in range(N):
        for i in sources -1: # Adjust for 0-based indexing in Python
            if CIJ[i, j] == 1:
                pths = np.concatenate((pths, np.array([[i+1, j+1]])), axis=1) # Adjust for 1-based indexing in MATLAB

    # Calculate use index for paths of length 1
    util[:, q-1] = np.histogram(pths.flatten(), bins=range(1, N + 2))[0] # Adjust for 1-based indexing
    #Enter paths of length 1 into Pq
    for np in range(pths.shape[1]):
        Pq[pths[0, np] - 1, pths[1, np] -1, q - 1] += 1 # Adjust for 0-based indexing

    # Begin saving all paths
    if savepths == 1:
        allpths = pths
    else:
        allpths = np.array([]).reshape(2,0)

    # Initialize
    npthscnt = K

    # Big loop for all other path lengths 'q'
    for q in range(2, qmax + 1):
        print(f'current pathlength (q) = {q},   number of paths so far (up to q-1)= {np.sum(Pq)}')

        # Estimate needed allocation for new paths
        len_npths = min(int(np.ceil(1.1 * npthscnt * K / N)), 100000000)
        npths = np.zeros((q + 1, len_npths))

        # Find unique endpoints of 'pths'
        endp = np.unique(pths[1, :])
        npthscnt = 0

        for ii in range(len(endp)):
            i = endp[ii] -1 #Adjust for 0-based indexing
            pa, pb = np.where(pths[1, :] == i + 1) #Adjust for 1-based indexing
            nendp = np.where(CIJ[i, :] == 1)[0]
            if len(nendp) > 0:
                for jj in range(len(nendp)):
                    j = nendp[jj]
                    pb_temp = pb[np.sum(j +1== pths[1:q, pb] +1, axis=0) == 0] # Adjust for 1-based indexing
                    npths[:, npthscnt:npthscnt + len(pb_temp)] = np.concatenate((pths[:, pb_temp], np.tile(np.array([j+1]), (1, len(pb_temp)))), axis=0) # Adjust for 1-based indexing
                    npthscnt += len(pb_temp)
                    Pq[:, j, q -1] += np.histogram(pths[0, pb_temp], bins=range(1, N + 2))[0] #Adjust for 1-based indexing


        if len_npths > npthscnt:
            npths = npths[:, :npthscnt]

        if savepths == 1:
            allpths = np.concatenate((allpths, np.zeros((2, allpths.shape[1]))), axis=1)
            allpths = np.concatenate((allpths, npths), axis=1)

        util[:, q - 1] += np.histogram(npths.flatten(), bins=range(1, N + 2))[0] - np.diag(Pq[:, :, q - 1])
        if npths.size > 0:
            pths = npths[:, npths[0, :] != npths[q, :]]
        else:
            pths = np.array([]).reshape(2,0)

        if pths.size == 0:
            qstop = q
            tpath = np.sum(Pq)
            plq = np.sum(Pq, axis=(0, 1))
            return Pq, tpath, plq, qstop, allpths, util

    qstop = q
    tpath = np.sum(Pq)
    plq = np.sum(Pq, axis=(0, 1))
    return Pq, tpath, plq, qstop, allpths, util

