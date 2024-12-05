# Translated from core_periphery_dir.m

import numpy as np

def core_periphery_dir(W, gamm=1, C=None):
    """
    Core/periphery structure and core-ness statistic.

    Args:
        W (numpy.ndarray): Directed (weighted or binary) connection matrix.
        gamm (float, optional): Core-ness resolution parameter. 
                                 gamma>1 detects small core/large periphery.
                                 0<=gamma<1 detects large core/small periphery. 
                                 Defaults to 1.
        C (numpy.ndarray, optional): Initial core structure (binary vector). 
                                     Defaults to a random binary vector.

    Returns:
        tuple: C (numpy.ndarray): Binary vector of optimal core structure 
                                   (1 for core, 0 for periphery).
               q (float): Maximized core-ness statistic.

    Algorithm: A version of the Kernighan-Lin algorithm for graph partitioning
               used in community detection (Newman, 2006) applied to optimize a
               core-structure objective described in Borgatti and Everett (2000).
    """
    n = len(W)  # Number of nodes
    W = np.array(W, dtype=float)  # Convert to float
    np.fill_diagonal(W, 0)  # Clear diagonal

    if C is None:
        C = np.random.rand(1, n) < 0.5
    else:
        C = C.reshape(1, n).astype(bool)

    s = np.sum(W)
    p = np.mean(W)
    b = W - gamm * p
    B = (b + b.T) / (2 * s)  # Directed core-ness matrix
    q = np.sum(B[C, C]) - np.sum(B[~C, ~C])  # Core-ness statistic

    f = 1  # Loop flag
    while f:
        f = 0
        Idx = np.arange(n)  # Initial node indices
        Ct = np.copy(C)
        while Idx.size > 0:
            Qt = np.zeros(n)  # Check swaps of node indices
            q0 = np.sum(B[Ct, Ct]) - np.sum(B[~Ct, ~Ct])
            Qt[Ct] = q0 - 2 * np.sum(B[Ct, :], axis=1)
            Qt[~Ct] = q0 + 2 * np.sum(B[~Ct, :], axis=1)

            max_Qt = np.max(Qt[Idx])  # Make swap with maximal increase in core-ness
            u = np.where(np.isclose(Qt[Idx], max_Qt))[0]
            u = u[np.random.randint(len(u))]
            Ct[Idx[u]] = ~Ct[Idx[u]]
            Idx = np.delete(Idx, u)

            if max_Qt - q > 1e-10:  # Recompute core-ness statistic
                f = 1
                C = Ct
                q = np.sum(B[C, C]) - np.sum(B[~C, ~C])
    
    q = np.sum(B[C, C]) - np.sum(B[~C, ~C])  # Return core-ness statistic
    return C.astype(int), q


