# Translated from null_model_dir_sign.m

import numpy as np

def null_model_dir_sign(W, bin_swaps=5, wei_freq=0.1):
    """
    Directed random graphs with preserved weight, degree and strength distributions.

    This function randomizes a directed network with positive and negative weights, while preserving the 
    degree and strength distributions. This function calls randmio_dir_signed (assumed to be defined elsewhere).

    Args:
        W (numpy.ndarray): Directed weighted connection matrix.
        bin_swaps (int, optional): Average number of swaps of each edge in binary randomization. 
                                    bin_swap=5 is the default (each edge rewired 5 times). 
                                    bin_swap=0 implies no binary randomization. Defaults to 5.
        wei_freq (float, optional): Frequency of weight sorting in weighted randomization. 
                                     wei_freq must be in the range of: 0 < wei_freq <= 1. 
                                     wei_freq=1 implies that weights are sorted at each step 
                                     (default in older versions of MATLAB). 
                                     wei_freq=0.1 implies that weights are sorted at each 10th step 
                                     (faster, default in newer versions of MATLAB). Defaults to 0.1.

    Returns:
        tuple: W0 (numpy.ndarray): Randomized weighted connection matrix.
               R (numpy.ndarray): Correlation coefficients between strength sequences of input and 
                                  output connection matrices.
    """

    if wei_freq <= 0 or wei_freq > 1:
        raise ValueError('wei_freq must be in the range of: 0 < wei_freq <= 1.')

    n = W.shape[0]  # number of nodes
    np.fill_diagonal(W, 0)  # clear diagonal
    Ap = W > 0  # positive adjacency matrix
    An = W < 0  # negative adjacency matrix

    # Assume randmio_dir_signed is defined elsewhere
    if np.sum(Ap) < (n * (n - 1)):  # if Ap is not full
        W_r = randmio_dir_signed(W, bin_swaps) # Assume randmio_dir_signed is defined elsewhere
        Ap_r = W_r > 0
        An_r = W_r < 0
    else:
        Ap_r = Ap
        An_r = An

    W0 = np.zeros((n, n))  # null model network

    for s in [1, -1]:
        if s == 1:
            Si = np.sum(W * Ap, axis=1)  # positive in-strength
            So = np.sum(W * Ap, axis=0)  # positive out-strength
            Wv = np.sort(W[Ap])  # sorted weights vector
            I, J = np.nonzero(Ap_r)  # weights indices
            Lij = n * (J) + I  # linear weights indices
        else:
            Si = np.sum(-W * An, axis=1)  # negative in-strength
            So = np.sum(-W * An, axis=0)  # negative out-strength
            Wv = np.sort(-W[An])  # sorted weights vector
            I, J = np.nonzero(An_r)  # weights indices
            Lij = n * (J) + I  # linear weights indices

        P = np.outer(So, Si)  # expected weights matrix

        if wei_freq == 1:
            for m in range(len(Wv), 0, -1):  # iteratively explore all weights
                Oind = np.argsort(P[Lij])  # get indices of Lij that sort P
                r = np.ceil(np.random.rand() * m).astype(int) -1
                o = Oind[r]  # choose random index of sorted expected weight
                W0.reshape(-1)[Lij[o]] = s * Wv[r]  # assign corresponding sorted weight at this index

                f = 1 - Wv[r] / So[I[o]]  # readjust expected weight probabilities for node I(o)
                P[I[o], :] = P[I[o], :] * f  # [1 - Wv(r)/S(I(o)) = (S(I(o)) - Wv(r))/S(I(o))]
                f = 1 - Wv[r] / Si[J[o]]  # readjust expected weight probabilities for node J(o)
                P[:, J[o]] = P[:, J[o]] * f  # [1 - Wv(r)/S(J(o)) = (S(J(o)) - Wv(r))/S(J(o))]

                So[I[o]] = So[I[o]] - Wv[r]  # readjust in-strength of node I(o)
                Si[J[o]] = Si[J[o]] - Wv[r]  # readjust out-strength of node J(o)
                Lij = np.delete(Lij, o)  # remove current index from further consideration
                I = np.delete(I, o)
                J = np.delete(J, o)
                Wv = np.delete(Wv, r)  # remove current weight from further consideration
        else:
            wei_period = int(np.round(1 / wei_freq))  # convert frequency to period
            for m in range(len(Wv), 0, -wei_period):  # iteratively explore at the given period
                Oind = np.argsort(P[Lij])  # get indices of Lij that sort P
                R = np.random.choice(m, min(m, wei_period), replace=False)
                O = Oind[R]  # choose random index of sorted expected weight
                W0.reshape(-1)[Lij[O]] = s * Wv[R]  # assign corresponding sorted weight at this index

                WAi = np.bincount(I[O], weights=Wv[R], minlength=n)
                Iu = WAi > 0
                F = 1 - WAi[Iu] / So[Iu]  # readjust expected weight probabilities for node I(o)
                P[Iu, :] = P[Iu, :] * F[:, np.newaxis]  # [1 - Wv(r)/S(I(o)) = (S(I(o)) - Wv(r))/S(I(o))]
                So[Iu] = So[Iu] - WAi[Iu]  # readjust in-strength of node I(o)

                WAj = np.bincount(J[O], weights=Wv[R], minlength=n)
                Ju = WAj > 0
                F = 1 - WAj[Ju] / Si[Ju]  # readjust expected weight probabilities for node J(o)
                P[:, Ju] = P[:, Ju] * F[:, np.newaxis]  # [1 - Wv(r)/S(J(o)) = (S(J(o)) - Wv(r))/S(J(o))]
                Si[Ju] = Si[Ju] - WAj[Ju]  # readjust out-strength of node J(o)

                Lij = np.delete(Lij, O)  # remove current index from further consideration
                I = np.delete(I, O)
                J = np.delete(J, O)
                Wv = np.delete(Wv, R)  # remove current weight from further consideration

    rpos_in = np.corrcoef(np.sum(W * (W > 0), axis=1), np.sum(W0 * (W0 > 0), axis=1))[0,1]
    rpos_ou = np.corrcoef(np.sum(W * (W > 0), axis=0), np.sum(W0 * (W0 > 0), axis=0))[0,1]
    rneg_in = np.corrcoef(np.sum(-W * (W < 0), axis=1), np.sum(-W0 * (W0 < 0), axis=1))[0,1]
    rneg_ou = np.corrcoef(np.sum(-W * (W < 0), axis=0), np.sum(-W0 * (W0 < 0), axis=0))[0,1]
    R = np.array([rpos_in, rpos_ou, rneg_in, rneg_ou])

    return W0, R

# Placeholder for randmio_dir_signed.  Replace with actual implementation if available.
def randmio_dir_signed(W, iter):
    return W


