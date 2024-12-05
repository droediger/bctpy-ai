# Translated from adjacency_plot_und.m

import numpy as np

def adjacency_plot_und(aij, coor=None):
    """Quick visualization tool

    Args:
        aij (numpy.ndarray): Adjacency matrix.
        coor (numpy.ndarray, optional): Node spatial coordinates. Defaults to None.

    Returns:
        tuple: Three vectors (X, Y, Z) that can be used for plotting the edges in aij.
    """
    n = len(aij)
    if coor is None:
        coor = np.zeros((n, 2))
        for i in range(n):
            coor[i, :] = [np.cos(2 * np.pi * (i) / n), np.sin(2 * np.pi * (i) / n)]
    
    i, j = np.nonzero(np.triu(aij, 1))
    _, p = np.sort(np.maximum(i,j))
    i = i[p]
    j = j[p]

    X = np.vstack((coor[i, 0], coor[j, 0]))
    Y = np.vstack((coor[i, 1], coor[j, 1]))
    if coor.shape[1] == 3:
        Z = np.vstack((coor[i, 2], coor[j, 2]))

    if isinstance(coor[0,0],(np.floating, float)) or not np.isnan(np.sum(coor)):
        X = np.concatenate((X, np.full((len(i),1), np.nan)))
        Y = np.concatenate((Y, np.full((len(i),1), np.nan)))
        if coor.shape[1] == 3:
            Z = np.concatenate((Z, np.full((len(i),1), np.nan)))

    X = X.flatten()
    Y = Y.flatten()
    if coor.shape[1] == 3:
        Z = Z.flatten()

    if coor.shape[1] == 3:
        return X, Y, Z
    else:
        return X, Y


