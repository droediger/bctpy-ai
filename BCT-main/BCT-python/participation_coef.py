# Translated from participation_coef.m

import numpy as np

def participation_coef(W, Ci, flag=0):
    """
    Participation coefficient.

    Calculates the participation coefficient, a measure of the diversity of intermodular connections of individual nodes.

    Args:
        W (numpy.ndarray): Binary/weighted, directed/undirected connection matrix.
        Ci (numpy.ndarray): Community affiliation vector.
        flag (int, optional): 0 for undirected graph (default), 1 for directed graph (out-degree), 2 for directed graph (in-degree).

    Returns:
        numpy.ndarray: Participation coefficient.
    """

    # Handle missing flag input
    if flag is None:
        flag = 0

    # Handle directed graph cases by transposing the adjacency matrix if necessary
    if flag == 2:
        W = W.T

    n = len(W)  # Number of vertices
    Ko = np.sum(W, axis=1)  # Degree (out-degree for directed graphs)
    Gc = (W != 0) * np.diag(Ci)  # Neighbor community affiliation
    Kc2 = np.zeros((n, 1))  # Community-specific neighbors

    # Compute community-specific neighbors' contribution to participation coefficient
    for i in range(1, int(np.max(Ci)) + 1):
        Kc2 += (np.sum(W * (Gc == i), axis=1) ** 2).reshape(-1,1)

    # Compute participation coefficient
    P = 1 - Kc2 / (Ko ** 2)
    P[Ko == 0] = 0  # P=0 for nodes with no (out)neighbors

    return P.flatten()


