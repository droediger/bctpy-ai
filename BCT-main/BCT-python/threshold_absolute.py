# Translated from threshold_absolute.m

import numpy as np

def threshold_absolute(W, thr):
    # THRESHOLD_ABSOLUTE    Absolute thresholding
    # 
    #   W_thr = threshold_absolute(W, thr);
    #
    #   This function thresholds the connectivity matrix by absolute weight
    #   magnitude. All weights below the given threshold, and all weights
    #   on the main diagonal (self-self connections) are set to 0.
    #
    #   Inputs: W           weighted or binary connectivity matrix
    #           thr         weight threshold
    #
    #   Output: W_thr       thresholded connectivity matrix
    #
    #
    #   Adapted from Mika Rubinov's MATLAB code, UNSW, 2009-2010

    np.fill_diagonal(W, 0)          #clear diagonal
    W[np.abs(W) < thr] = 0          #apply threshold

    return W


