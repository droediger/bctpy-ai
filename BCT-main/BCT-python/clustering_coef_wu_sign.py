# Translated from clustering_coef_wu_sign.m

import numpy as np

def clustering_coef_wu_sign(W, coef_type=1):
    """
    Multiple generalizations of the clustering coefficient.

    Parameters
    ----------
    W : numpy.ndarray
        Weighted undirected connection matrix.
    coef_type : int, optional
        Desired type of clustering coefficient. Options:
        1 (default): Onnela et al. formula. Computed separately for positive & negative weights.
        2: Zhang & Horvath formula. Computed separately for positive & negative weights.
        3: Constantini & Perugini's generalization. Takes both positive & negative weights into account simultaneously. Produces only one value.

    Returns
    -------
    C_pos : numpy.ndarray
        Clustering coefficient vector for positive weights or the single vector for coef_type=3.
    C_neg : numpy.ndarray
        Clustering coefficient vector for negative weights (not returned for coef_type=3).
    Ctot_pos : float
        Mean clustering coefficient for positive weights.
    Ctot_neg : float
        Mean clustering coefficient for negative weights (not returned for coef_type=3).

    """
    n = len(W)  # Number of nodes
    np.fill_diagonal(W, 0)  # Set diagonal to zero

    if coef_type == 1:
        W_pos = np.multiply(W, W > 0)
        K_pos = np.sum(W_pos != 0, axis=1)
        cyc3_pos = np.diag(np.linalg.matrix_power(W_pos, 3))
        K_pos[cyc3_pos == 0] = np.inf  # If no 3-cycles exist, make C=0 (via K=inf)
        C_pos = np.divide(cyc3_pos, K_pos * (K_pos - 1))  # Clustering coefficient
        Ctot_pos = np.mean(C_pos)

        W_neg = -np.multiply(W, W < 0)
        K_neg = np.sum(W_neg != 0, axis=1)
        cyc3_neg = np.diag(np.linalg.matrix_power(W_neg, 3))
        K_neg[cyc3_neg == 0] = np.inf  # If no 3-cycles exist, make C=0 (via K=inf)
        C_neg = np.divide(cyc3_neg, K_neg * (K_neg - 1))  # Clustering coefficient
        Ctot_neg = np.mean(C_neg)
    elif coef_type == 2:
        W_pos = np.multiply(W, W > 0)
        cyc3_pos = np.zeros(n)
        cyc2_pos = np.zeros(n)
        for i in range(n):
            for j in range(n):
                for q in range(n):
                    cyc3_pos[i] += W_pos[j, i] * W_pos[i, q] * W_pos[j, q]
                    if j != q:
                        cyc2_pos[i] += W_pos[j, i] * W_pos[i, q]
        cyc2_pos[cyc3_pos == 0] = np.inf  # If no 3-cycles exist, make C=0 (via K=inf)
        C_pos = np.divide(cyc3_pos, cyc2_pos)  # Clustering coefficient
        Ctot_pos = np.mean(C_pos)

        W_neg = -np.multiply(W, W < 0)
        cyc3_neg = np.zeros(n)
        cyc2_neg = np.zeros(n)
        for i in range(n):
            for j in range(n):
                for q in range(n):
                    cyc3_neg[i] += W_neg[j, i] * W_neg[i, q] * W_neg[j, q]
                    if j != q:
                        cyc2_neg[i] += W_neg[j, i] * W_neg[i, q]
        cyc2_neg[cyc3_neg == 0] = np.inf  # If no 3-cycles exist, make C=0 (via K=inf)
        C_neg = np.divide(cyc3_neg, cyc2_neg)  # Clustering coefficient
        Ctot_neg = np.mean(C_neg)
    elif coef_type == 3:
        cyc3 = np.zeros(n)
        cyc2 = np.zeros(n)
        for i in range(n):
            for j in range(n):
                for q in range(n):
                    cyc3[i] += W[j, i] * W[i, q] * W[j, q]
                    if j != q:
                        cyc2[i] += np.abs(W[j, i] * W[i, q])
        cyc2[cyc3 == 0] = np.inf  # If no 3-cycles exist, make C=0 (via K=inf)
        C_pos = np.divide(cyc3, cyc2)  # Clustering coefficient
        Ctot_pos = np.mean(C_pos)
        C_neg = np.full(len(C_pos), np.nan)
        Ctot_neg = np.nan

    return C_pos, C_neg, Ctot_pos, Ctot_neg

