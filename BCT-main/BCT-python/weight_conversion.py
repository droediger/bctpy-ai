# Translated from weight_conversion.m

import numpy as np

def weight_conversion(W, wcm):
    # WEIGHT_CONVERSION    Conversion of weights in input matrix
    #
    #   W_bin = weight_conversion(W, 'binarize');
    #   W_nrm = weight_conversion(W, 'normalize');
    #   L = weight_conversion(W, 'lengths');
    #   W_fix = weight_conversion(W, 'autofix');
    #
    #   This function may either binarize an input weighted connection matrix,
    #   normalize an input weighted connection matrix, convert an input
    #   weighted connection matrix to a weighted connection-length matrix, or
    #   fix common connection problems in binary or weighted connection matrices.
    #
    #       Binarization converts all present connection weights to 1.
    #
    #       Normalization rescales all weight magnitudes to the range [0,1] and
    #   should be done prior to computing some weighted measures, such as the
    #   weighted clustering coefficient.
    #
    #       Conversion of connection weights to connection lengths is needed
    #   prior to computation of weighted distance-based measures, such as
    #   distance and betweenness centrality. In a weighted connection network,
    #   higher weights are naturally interpreted as shorter lengths. The
    #   connection-lengths matrix here is defined as the inverse of the
    #   connection-weights matrix.
    #
    #       Autofix removes all Inf and NaN values, remove all self connections
    #   (sets all weights on the main diagonal to 0), ensures that symmetric matrices
    #   are exactly symmetric (by correcting for round-off error), and ensures that
    #   binary matrices are exactly binary (by correcting for round-off error).
    #
    #   Inputs: W           binary or weighted connectivity matrix
    #           wcm         weight-conversion command - possible values:
    #                           'binarize'      binarize weights
    #                           'normalize'     normalize weights
    #                           'lengths'       convert weights to lengths
    #                           'autofix'       fixes common weights problems
    #
    #   Output: W_          output connectivity matrix
    #
    #
    #   Mika Rubinov, U Cambridge, 2012
    #
    #   Modification History:
    #   Sep 2012: Original
    #   Jan 2015: Added autofix feature.
    #   Jan 2017: Corrected bug in autofix (thanks to Jeff Spielberg)

    if wcm == 'binarize':
        W = (W != 0).astype(float)  # binarize
    elif wcm == 'normalize':
        W = W / np.max(np.abs(W))  # rescale by maximal weight
    elif wcm == 'lengths':
        E = np.nonzero(W)
        W[E] = 1.0 / W[E]  # invert weights
    elif wcm == 'autofix':
        # clear diagonal
        n = len(W)
        W[np.arange(n), np.arange(n)] = 0

        # remove Infs and NaNs
        idx = np.isnan(W) | np.isinf(W)
        if np.any(idx):
            W[idx] = 0

        # ensure exact binariness
        U = np.unique(W)
        if len(U) > 1:
            idx_0 = np.abs(W) < 1e-10
            idx_1 = np.abs(W - 1) < 1e-10
            if np.all(idx_0 | idx_1):
                W[idx_0] = 0
                W[idx_1] = 1

        # ensure exact symmetry
        if not np.array_equal(W, W.T):
            if np.max(np.abs(W - W.T)) < 1e-10:
                W = (W + W.T) / 2
    else:
        raise ValueError('Unknown weight-conversion command.')
    return W

