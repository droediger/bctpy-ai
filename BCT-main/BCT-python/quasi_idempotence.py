# Translated from quasi_idempotence.m

import numpy as np

def quasi_idempotence(X, K=np.inf):
    """
    Connection matrix quasi-idempotence

    Parameters
    ----------
    X : numpy.ndarray
        Undirected weighted/binary connection matrix with non-negative weights.
    K : int or float, optional
        Number of iterations. 
        K < np.inf: iterate a predetermined number of times.
        K = np.inf: iterate until convergence is attained (default).

    Returns
    -------
    XN : numpy.ndarray
        The final matrix, from the last iteration.
    IOTA : numpy.ndarray
        The vector of correlation coefficients, one per iteration.
    EPS : numpy.ndarray
        The vector of errors, one per iteration.
    U : int
        The number of iterations performed.

    Notes
    -----
    See Minati et al. (2017) for a discussion of the issues associated with applying this measure to binary graphs and the significance of IOTA[0] and IOTA[-1].
    """

    X = np.double(X) # enforce double-precision format
    if np.isnan(X).any(): # bail out if any nan found
        raise ValueError('found nan, giving up!')
    if (X < 0).any(): # bail out if any negative elements found
        raise ValueError('found a negative element, giving up!')
    
    N = X.shape[0] # get matrix size
    np.fill_diagonal(X, 0) # null the diagonal in the initial matrix
    X = X / np.linalg.norm(X) # normalize to unit norm
    XN = X
    mask = np.triu(np.ones((N,N), dtype=bool), 1) # create mask for superdiagonal elements
    U = 0 # initialize iterations counter
    IOTA = [] # this vector will contain the correlation coefficients
    EPS = np.inf # and this will contain the errors

    if np.isinf(K):
        while EPS[-1] > np.finfo(float).eps: # iterate until error below precision
            U += 1 # increase iteration counter
            XN_hat = XN # save the initial matrix
            XN = np.linalg.matrix_power(XN, 2) # square the matrix
            XN = XN / np.linalg.norm(XN) # normalize it again (for numerical reasons)
            IOTA.append(np.corrcoef(X[mask].flatten(), XN[mask].flatten())[0,1]) # calculate correlation coefficient
            EPS = np.append(EPS, np.linalg.norm(XN_hat - XN)) # calculate error
    else:
        while U < K: # iterate a prescribed number of times
            U += 1 # increase iteration counter
            XN_hat = XN # save the initial matrix
            XN = np.linalg.matrix_power(XN, 2) # square the matrix
            XN = XN / np.linalg.norm(XN) # normalize it again (for numerical reasons)
            IOTA.append(np.corrcoef(X[mask].flatten(), XN[mask].flatten())[0,1]) # calculate correlation coefficient
            EPS = np.append(EPS, np.linalg.norm(XN_hat - XN)) # calculate error

    EPS = EPS[1:]
    return XN, np.array(IOTA), np.array(EPS), U

