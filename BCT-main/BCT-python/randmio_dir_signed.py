# Translated from randmio_dir_signed.m

import numpy as np

def randmio_dir_signed(W, ITER):
    """
    Random directed graph with preserved signed in/out degree distribution.

    Parameters
    ----------
    W : array_like
        Directed (binary/weighted) connection matrix.
    ITER : int
        Rewiring parameter (each edge is rewired approximately ITER times).

    Returns
    -------
    R : array_like
        Randomized network.
    eff : int
        Number of actual rewirings carried out.

    References
    ----------
    Maslov and Sneppen (2002) Science 296:910
    """

    R = np.array(W, dtype=float)  # sign function requires float input
    n = R.shape[0]
    ITER = ITER * n * (n - 1)

    # Maximal number of rewiring attempts per 'iter'
    maxAttempts = n
    # Actual number of successful rewirings
    eff = 0

    for iter in range(ITER):
        att = 0
        while att <= maxAttempts:  # while not rewired
            # select four distinct vertices
            nodes = np.random.choice(n, 4, replace=False)
            a, b, c, d = nodes

            r0_ab = R[a, b]
            r0_cd = R[c, d]
            r0_ad = R[a, d]
            r0_cb = R[c, b]

            # rewiring condition
            if (np.sign(r0_ab) == np.sign(r0_cd)) and \
               (np.sign(r0_ad) == np.sign(r0_cb)) and \
               (np.sign(r0_ab) != np.sign(r0_ad)):

                R[a, d] = r0_ab
                R[a, b] = r0_ad
                R[c, b] = r0_cd
                R[c, d] = r0_cb

                eff += 1
                break
            att += 1
    return R, eff

