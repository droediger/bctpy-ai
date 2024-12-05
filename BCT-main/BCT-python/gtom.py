# Translated from gtom.m

import numpy as np

def gtom(adj, numSteps):
    """
    Generalized topological overlap measure.

    Computes the M x M generalized topological overlap measure (GTOM) matrix for a given number of steps.

    Args:
        adj: Adjacency matrix (binary, undirected).
        numSteps: Number of steps.

    Returns:
        gt: GTOM matrix.
    """

    #initial state for bm matrix
    bm = np.copy(adj)
    bmAux = np.copy(bm)
    numNodes = adj.shape[0]

    if numSteps > numNodes:
        print('warning, reached maximum value for numSteps. numSteps reduced to adj-size')
        numSteps = numNodes

    if numSteps == 0:
        # GTOM0
        gt = adj
    else:
        for steps in range(2, numSteps + 1):
            for i in range(numNodes):
                #neighbors of node i
                neighColumn = np.where(bm[i, :] == 1)[0]

                #neighbors of neighbors of node i
                neighNeighColumn = np.where(bm[neighColumn, :] == 1)[0]
                newNeigh = np.setdiff1d(np.unique(neighNeighColumn), i)

                #neighbors of neighbors of node i become considered node i neighbors
                bmAux[i, newNeigh] = 1
                #keep symmetry of matrix
                bmAux[newNeigh, i] = 1

            #bm is updated with new step all at once
            bm = np.copy(bmAux)

        #clear bmAux newNeigh; these are not needed after the loop

        #numerators of GTOM formula
        numeratorMatrix = bm @ bm + adj + np.eye(numNodes)

        #vector containing degree of each node
        bmSum = np.sum(bm, axis=1)
        #clear bm; this is not needed after the sum

        denominatorMatrix = -adj + np.minimum(np.tile(bmSum, (numNodes, 1)), np.tile(bmSum[:, np.newaxis], (1, numNodes))) + 1
        gt = numeratorMatrix / denominatorMatrix

    return gt

