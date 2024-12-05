# Translated from flow_coef_bd.m

import numpy as np

def flow_coef_bd(CIJ):
    """
    Node-wise flow coefficients

    Computes the flow coefficient for each node and averaged over the
    network, as described in Honey et al. (2007) PNAS. The flow coefficient
    is similar to betweenness centrality, but works on a local
    neighborhood. It is mathematically related to the clustering
    coefficient  (cc) at each node as, fc+cc <= 1.

    Args:
        CIJ (numpy.ndarray): connection/adjacency matrix (binary, directed)

    Returns:
        tuple: fc (numpy.ndarray): flow coefficient for each node
               FC (float): average flow coefficient over the network
               total_flo (numpy.ndarray): number of paths that "flow" across the central node

    Reference:  Honey et al. (2007) Proc Natl Acad Sci U S A

    """
    N = CIJ.shape[0]

    # Initialize
    fc = np.zeros(N)
    total_flo = np.zeros(N)
    max_flo = np.zeros(N)

    # Loop over nodes
    for v in range(N):
        # Find neighbors - note: treats incoming and outgoing connections as equal
        nb = np.where((CIJ[v, :] + CIJ[:, v].T) > 0)[0]
        if len(nb) > 0:
            CIJflo = -CIJ[np.ix_(nb, nb)]
            for i in range(len(nb)):
                for j in range(len(nb)):
                    if CIJ[nb[i], v] == 1 and CIJ[v, nb[j]] == 1:
                        CIJflo[i, j] += 1
            total_flo[v] = np.sum((CIJflo == 1).astype(int) * (1 - np.eye(len(nb))))
            max_flo[v] = len(nb)**2 - len(nb)
            fc[v] = total_flo[v] / max_flo[v]

    # Handle nodes that are NaNs
    fc[np.isnan(fc)] = 0

    FC = np.mean(fc)
    return fc, FC, total_flo


