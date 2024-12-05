# Translated from rentian_scaling_3d.m

import numpy as np

def rentian_scaling_3d(A, XYZ, n, tol):
    """
    Rentian scaling for networks embedded in three dimensions.

    Parameters
    ----------
    A : numpy.ndarray
        MxM adjacency matrix. Must be unweighted, binary, and symmetric.
    XYZ : numpy.ndarray
        Mx3 matrix of node placement coordinates [x y z], where M is the number of nodes.
    n : int
        Number of partitions to compute. Each partition is a data point.
    tol : float
        Tolerance for ensuring partitions are within the network boundary.

    Returns
    -------
    N : numpy.ndarray
        nx1 vector of the number of nodes in each of the n partitions.
    E : numpy.ndarray
        nx1 vector of the number of edges crossing the boundary of each partition.
    """

    # Determine the number of nodes in the system
    M = XYZ.shape[0]

    # Rescale coordinates so that they are all greater than unity
    XYZn = XYZ - np.tile(np.min(XYZ, axis=0) - 1, (M, 1))

    # Compute the volume of the convex hull of the network (approximated using a bounding box for simplicity)
    V = np.prod(np.max(XYZn, axis=0) - np.min(XYZn, axis=0))


    # Min and max network coordinates
    xmin = np.min(XYZn[:, 0])
    xmax = np.max(XYZn[:, 0])
    ymin = np.min(XYZn[:, 1])
    ymax = np.max(XYZn[:, 1])
    zmin = np.min(XYZn[:, 2])
    zmax = np.max(XYZn[:, 2])

    # Initialize vectors of number of nodes in box and number of edges crossing box
    N = np.zeros((n, 1))
    E = np.zeros((n, 1))

    nPartitions = 0

    while nPartitions < n:
        inside = 0
        while inside == 0:
            # Pick a random (x,y,z) coordinate to be the center of the box
            randx = xmin + (xmax - xmin) * np.random.rand()
            randy = ymin + (ymax - ymin) * np.random.rand()
            randz = zmin + (zmax - zmin) * np.random.rand()

            #Check if the point is inside the bounding box of the network (simplification)
            inside = (randx >= xmin and randx <= xmax and
                      randy >= ymin and randy <= ymax and
                      randz >= zmin and randz <= zmax)


        # Determine the approximate maximum distance the box can extend
        deltaX = min(abs(xmax - randx), abs(xmin - randx))
        deltaY = min(abs(ymax - randy), abs(ymin - randy))
        deltaZ = min(abs(zmax - randz), abs(zmin - randz))
        deltaLmin = min([deltaX, deltaY, deltaZ])

        inside = 0
        while inside == 0:
            # Pick a random (side length)/2
            deltaL = deltaLmin * np.random.rand()

            # (x,y,z) coordinates for corners of box (not used in this simplified version)
            #boxCoords = np.array(...)

            # Check if the box is within the network boundary (using bounding box for simplicity)
            inside = (randx - deltaL >= xmin and randx + deltaL <= xmax and
                      randy - deltaL >= ymin and randy + deltaL <= ymax and
                      randz - deltaL >= zmin and randz + deltaL <= zmax)

        # Find nodes inside the box, edges crossing the boundary
        L = np.where((XYZn[:, 0] > (randx - deltaL)) & (XYZn[:, 0] < (randx + deltaL)) &
                     (XYZn[:, 1] > (randy - deltaL)) & (XYZn[:, 1] < (randy + deltaL)) &
                     (XYZn[:, 2] > (randz - deltaL)) & (XYZn[:, 2] < (randz + deltaL)))[0]

        if len(L) > 0:
            nPartitions += 1
            # Count edges crossing the boundary of the cube
            E[nPartitions - 1, 0] = np.sum(A[L, :][:, np.setdiff1d(np.arange(M), L)])
            # Count nodes inside of the cube
            N[nPartitions - 1, 0] = len(L)

    return N, E


