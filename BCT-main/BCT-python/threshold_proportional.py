# Translated from threshold_proportional.m

import numpy as np

def threshold_proportional(W, p):
    #THRESHOLD_PROPORTIONAL     Proportional thresholding
    #
    #   W_thr = threshold_proportional(W, p);
    #
    #   This function "thresholds" the connectivity matrix by preserving a
    #   proportion p (0<p<1) of the strongest weights. All other weights, and
    #   all weights on the main diagonal (self-self connections) are set to 0.
    #
    #   Inputs: W,      weighted or binary connectivity matrix
    #           p,      proportion of weights to preserve
    #                       range:  p=1 (all weights preserved) to
    #                               p=0 (no weights preserved)
    #
    #   Output: W_thr,  thresholded connectivity matrix
    #
    #
    #   Mika Rubinov, U Cambridge,
    #   Roan LaPlante, Martinos Center, MGH
    #   Zitong Zhang, Penn Engineering

    #   Modification history:
    #   2010: Original (MR)
    #   2012: Bug fix for symmetric matrices (RLP)
    #   2015: Improved symmetricity test (ZZ)

    n = W.shape[0]                               #number of nodes
    np.fill_diagonal(W, 0)                             #clear diagonal

    if np.max(np.abs(W - W.T)) < 1e-10:             #if symmetric matrix
        W = np.triu(W)                              #ensure symmetry is preserved
        ud = 2                                   #halve number of removed links
    else:
        ud = 1

    ind = np.nonzero(W)                               #find all links
    E = np.array(sorted(zip(ind[0]*W.shape[1]+ind[1],W[ind]), key=lambda x: x[1], reverse=True))               #sort by magnitude
    en = int(round((n**2-n)*p/ud))                     #number of links to be preserved

    W[E[en:,0]//n, E[en:,0]%n] = 0                         #apply threshold

    if ud == 2:                                    #if symmetric matrix
        W = W + W.T                                #reconstruct symmetry

    return W

