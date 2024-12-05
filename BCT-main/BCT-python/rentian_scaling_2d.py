# Translated from rentian_scaling_2d.m

import numpy as np

def rentian_scaling_2d(A, XY, n, tol):
    """
    Rentian scaling for networks embedded in two dimensions.

    Parameters
    ----------
    A : numpy.ndarray
        MxM adjacency matrix. Must be unweighted, binary, and symmetric.
    XY : numpy.ndarray
        Matrix of node placement coordinates. Must be in the form of an Mx2 
        matrix [x y], where M is the number of nodes and x and y are column 
        vectors of node coordinates.
    n : int
        Number of partitions to compute. Each partition is a data point. You 
        want a large enough number to adequately estimate the Rent's exponent.
    tol : float
        This should be a small value (for example 1e-6). In order to mitigate 
        the effects of boundary conditions due to the finite size of the 
        network, we only allow partitions that are contained within the 
        boundary of the network. This is achieved by first computing the 
        volume of the convex hull of the node coordinates (V). We then ensure 
        that the volume of the convex hull computed on the original node 
        coordinates plus the coordinates of the randomly generated partition 
        (Vnew) is within a given tolerance of the original (i.e. check 
        abs(V - Vnew) < tol). Thus tol, should be a small value in order to 
        make sure the partitions are contained largely within the boundary of 
        the network, and thus the number of nodes and edges within the box are 
        not skewed by finite size effects.

    Returns
    -------
    N : numpy.ndarray
        nx1 vector of the number of nodes in each of the n partitions.
    E : numpy.ndarray
        nx1 vector of the number of edges crossing the boundary of each 
        partition.
    """

    # Determine the number of nodes in the system
    M = XY.shape[0]

    # Rescale coordinates so that they are all greater than unity
    XYn = XY - np.tile(np.min(XY, axis=0) - 1, (M, 1))

    # Compute the area of convex hull (i.e., area of the boundary) of the network
    from scipy.spatial import ConvexHull
    hull = ConvexHull(XYn)
    V = hull.volume

    # Min and max network coordinates
    xmin = np.min(XYn[:, 0])
    xmax = np.max(XYn[:, 0])
    ymin = np.min(XYn[:, 1])
    ymax = np.max(XYn[:, 1])

    # Initialize vectors of number of nodes in box and number of edges crossing box
    N = np.zeros((n, 1))
    E = np.zeros((n, 1))

    # Create partitions, and count the number of nodes inside the partition (N) and the number of edges traversing the boundary of the partition (E)
    nPartitions = 0
    while nPartitions < n:
        inside = 0
        while inside == 0:
            # Pick a random (x,y) coordinate to be the center of the box
            randx = xmin + (xmax - xmin) * np.random.rand()
            randy = ymin + (ymax - ymin) * np.random.rand()

            # Make sure the point is inside the convex hull of the network
            newCoords = np.vstack((XYn, np.array([randx, randy])))
            hull = ConvexHull(newCoords)
            Vnew = hull.volume

            # If the old convex hull area and new convex hull area are equal then the box center must be inside the network boundary.
            if np.isclose(V, Vnew, atol=1e-10):  #Using np.isclose for floating-point comparison
                inside = 1

        # Determine the approximate maximum distance the box can extend, given the center point and the bounds of the network
        deltaY = min(abs(ymax - randy), abs(ymin - randy))
        deltaX = min(abs(xmax - randx), abs(xmin - randx))
        deltaLmin = min(deltaY, deltaX)

        inside = 0
        while inside == 0:
            # Pick a random (side length)/2 that is between 0 and the max possible
            deltaL = deltaLmin * np.random.rand()

            # (x,y) coordinates for corners of box
            boxCoords = np.array([
                [randx - deltaL, randy - deltaL],
                [randx - deltaL, randy + deltaL],
                [randx + deltaL, randy - deltaL],
                [randx + deltaL, randy + deltaL]
            ])

            # Check if all corners of box are inside the convex hull of the network
            newCoords = np.vstack((XYn, boxCoords))
            hull = ConvexHull(newCoords)
            Vnew = hull.volume

            # Make sure the new convex hull that includes the partition corners is within a certain tolerance of the original convex hull area.
            if abs(V - Vnew) <= tol:
                inside = 1

        # Find nodes inside the box, edges crossing the boundary
        L = np.where((XYn[:, 0] > (randx - deltaL)) & (XYn[:, 0] < (randx + deltaL)) &
                     (XYn[:, 1] > (randy - deltaL)) & (XYn[:, 1] < (randy + deltaL)))[0]

        if len(L) > 0:
            nPartitions += 1
            # Count edges crossing the boundary of the box
            E[nPartitions - 1, 0] = np.sum(A[L, :][:, np.setdiff1d(np.arange(M), L)])
            # Count nodes inside of the box
            N[nPartitions - 1, 0] = len(L)

    return N[:nPartitions], E[:nPartitions]

