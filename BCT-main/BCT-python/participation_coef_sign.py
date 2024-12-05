# Translated from participation_coef_sign.m

import numpy as np

def participation_coef_sign(W, Ci):
    """
    Participation coefficient.

    Parameters
    ----------
    W : array_like
        Undirected connection matrix with positive and negative weights.
    Ci : array_like
        Community affiliation vector.

    Returns
    -------
    Ppos : array_like
        Participation coefficient from positive weights.
    Pneg : array_like
        Participation coefficient from negative weights.

    References
    ----------
    Guimera R, Amaral L. Nature (2005) 433:895-900.
    """
    n = len(W)  # Number of vertices

    Ppos = pcoef(W * (W > 0))
    Pneg = pcoef(-W * (W < 0))

    return Ppos, Pneg


def pcoef(W_):
    S = np.sum(W_, axis=1)  # Strength
    Gc = (W_ != 0) * np.diag(Ci)  # Neighbor community affiliation
    Sc2 = np.zeros((len(W_), 1))  # Community-specific neighbors

    for i in range(1, int(np.max(Ci)) + 1):
        Sc2 = Sc2 + (np.sum(W_ * (Gc == i), axis=1)**2).reshape(-1,1)

    P = np.ones((len(W_), 1)) - Sc2 / (S.reshape(-1,1)**2)
    P = np.nan_to_num(P) #Handle NaN values
    P[P == 0] = 0  # p_ind=0 if no (out)neighbors

    return P.flatten()



