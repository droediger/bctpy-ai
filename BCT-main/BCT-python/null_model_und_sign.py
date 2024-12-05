# Translated from null_model_und_sign.m

import numpy as np

def null_model_und_sign(W, bin_swaps=5, wei_freq=0.1):
    """
    Randomizes an undirected network with positive and negative weights, preserving the degree and strength distributions.

    Args:
        W (numpy.ndarray): Undirected weighted connection matrix.
        bin_swaps (int, optional): Average number of swaps of each edge in binary randomization. Defaults to 5.
        wei_freq (float, optional): Frequency of weight sorting in weighted randomization. Defaults to 0.1.

    Returns:
        tuple: A tuple containing the randomized weighted connection matrix (W0) and the correlation coefficient between strength sequences of input and output connection matrices (R).
    """

    if wei_freq <= 0 or wei_freq > 1:
        raise ValueError('wei_freq must be in the range of: 0 < wei_freq <= 1.')

    n = W.shape[0]  # Number of nodes
    np.fill_diagonal(W, 0)  # Clear diagonal

    Ap = W > 0  # Positive adjacency matrix
    An = W < 0  # Negative adjacency matrix

    # Placeholder for randmio_und_signed function.  Assume it's defined elsewhere and works as expected.
    if np.sum(Ap) < n * (n - 1):  # If Ap is not full
        W_r = randmio_und_signed(W, bin_swaps) # Assume randmio_und_signed is defined elsewhere.
        Ap_r = W_r > 0
        An_r = W_r < 0
    else:
        Ap_r = Ap
        An_r = An

    W0 = np.zeros((n, n))  # Null model network

    for s in [1, -1]:  # Iterate over positive and negative weights
        if s == 1:
            S = np.sum(W * Ap, axis=1)  # Positive strength
            Wv = np.sort(W[np.triu(Ap, k=1)])  # Sorted weights vector, excluding diagonal
            I, J = np.nonzero(np.triu(Ap_r,k=1))  # Weights indices
            Lij = n * J + I  # Linear weights indices

        else:
            S = np.sum(-W * An, axis=1)  # Negative strength
            Wv = np.sort(-W[np.triu(An, k=1)])  # Sorted weights vector, excluding diagonal.
            I, J = np.nonzero(np.triu(An_r,k=1))  # Weights indices
            Lij = n * J + I  # Linear weights indices

        P = np.outer(S, S)  # Expected weights matrix

        if wei_freq == 1:
            for m in range(len(Wv), 0, -1):  # Iteratively explore all weights
                Oind = np.argsort(P.flatten()[Lij])  # Indices of Lij that sort P
                r = np.ceil(np.random.rand() * m) -1 # adjust index for  indexing
                o = Oind[int(r)]  # Choose random index of sorted expected weight
                W0.flatten()[Lij[o]] = s * Wv[int(r)]  # Assign corresponding sorted weight at this index

                f = 1 - Wv[int(r)] / S[I[o]]  # Readjust expected weight probabilities for node I(o)
                P[I[o], :] = P[I[o], :] * f
                P[:, I[o]] = P[:, I[o]] * f
                f = 1 - Wv[int(r)] / S[J[o]]  # Readjust expected weight probabilities for node J(o)
                P[J[o], :] = P[J[o], :] * f
                P[:, J[o]] = P[:, J[o]] * f

                S[[I[o], J[o]]] = S[[I[o], J[o]]] - Wv[int(r)]  # Readjust strengths of nodes I(o) and J(o)
                Lij = np.delete(Lij, o)  # Remove current index from further consideration
                I = np.delete(I, o)
                J = np.delete(J, o)
                Wv = np.delete(Wv, int(r))  # Remove current weight from further consideration
        else:
            wei_period = round(1 / wei_freq)  # Convert frequency to period
            for m in range(len(Wv), 0, -wei_period):  # Iteratively explore at the given period
                Oind = np.argsort(P.flatten()[Lij])  # Indices of Lij that sort P
                R = np.random.choice(m, min(m, wei_period), replace=False)
                O = Oind[R]
                W0.flatten()[Lij[O]] = s * Wv[R]  # Assign corresponding sorted weight at this index

                WA = np.zeros(n)
                np.add.at(WA, [I[O],J[O]], Wv[R])  #cumulative weight

                IJu = WA > 0
                F = 1 - WA[IJu] / S[IJu]
                F = np.tile(F,(n,1)).T
                P[IJu, :] = P[IJu, :] * F  # Readjust expected weight probabilities
                P[:, IJu] = P[:, IJu] * F.T
                S[IJu] = S[IJu] - WA[IJu]  # Re-adjust strengths

                Lij = np.delete(Lij, O)  # Remove current index from further consideration
                I = np.delete(I, O)
                J = np.delete(J, O)
                Wv = np.delete(Wv, R)  # Remove current weight from further consideration

    W0 = W0 + W0.T

    rpos = np.corrcoef(np.sum(W * (W > 0), axis=1), np.sum(W0 * (W0 > 0), axis=1))
    rneg = np.corrcoef(np.sum(-W * (W < 0), axis=1), np.sum(-W0 * (W0 < 0), axis=1))
    R = [rpos[0, 1], rneg[0, 1]]

    return W0, R

# Placeholder for randmio_und_signed;  replace with your actual implementation
def randmio_und_signed(W, iter):
    """Placeholder for randmio_und_signed function. Replace with your actual implementation."""
    return W


