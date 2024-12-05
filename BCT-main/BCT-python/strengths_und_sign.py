# Translated from strengths_und_sign.m

import numpy as np

def strengths_und_sign(W):
    """
    Calculates node strength and total weight for an undirected weighted network.

    Args:
        W: Undirected connection matrix with positive and negative weights.

    Returns:
        Spos: Nodal strength of positive weights.
        Sneg: Nodal strength of negative weights.
        vpos (optional): Total positive weight.
        vneg (optional): Total negative weight.

    """
    n = len(W)  # Number of nodes
    np.fill_diagonal(W, 0)  # Clear diagonal
    Spos = np.sum(W * (W > 0), axis=0)  # Positive strengths
    Sneg = np.sum(-W * (W < 0), axis=0)  # Negative strengths

    if n > 2:
        vpos = np.sum(Spos)  # Total positive weight
        vneg = np.sum(Sneg)  # Total negative weight
        return Spos, Sneg, vpos, vneg
    else:
        return Spos, Sneg



