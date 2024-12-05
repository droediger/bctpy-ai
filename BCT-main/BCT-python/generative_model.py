# Translated from generative_model.m

import numpy as np

def generative_model(A, D, m, modeltype, modelvar, params, epsilon=1e-5):
    """
    Run generative model code

    Generates synthetic networks using the models described in the study by
    Betzel et al (2016) in Neuroimage.

    Args:
        A (numpy.ndarray): binary network of seed connections
        D (numpy.ndarray): Euclidean distance/fiber length matrix
        m (int): number of connections that should be present in the final synthetic network
        modeltype (str): specifies the generative rule (see below)
        modelvar (list): specifies whether the generative rules are based on power-law or exponential relationship (['powerlaw'] | ['exponential'])
        params (numpy.ndarray): either a vector (in the case of the geometric model) or a matrix (for all other models) of parameters at which the model should be evaluated.
        epsilon (float, optional): the baseline probability of forming a particular connection (should be a very small number). Defaults to 1e-5.

    Returns:
        numpy.ndarray: m x number of networks matrix of connections


    Full list of model types:
    (each model type realizes a different generative rule)

        1.  'sptl'          spatial model
        2.  'neighbors'     number of common neighbors
        3.  'matching'      matching index
        4.  'clu-avg'       average clustering coeff.
        5.  'clu-min'       minimum clustering coeff.
        6.  'clu-max'       maximum clustering coeff.
        7.  'clu-diff'      difference in clustering coeff.
        8.  'clu-prod'      product of clustering coeff.
        9.  'deg-avg'       average degree
        10. 'deg-min'       minimum degree
        11. 'deg-max'       maximum degree
        12. 'deg-diff'      difference in degree
        13. 'deg-prod'      product of degree
    """
    n = len(D)
    nparams = params.shape[0]
    b = np.zeros((m, nparams))

    if modeltype == 'clu-avg':
        clu = clustering_coef_bu(A)
        Kseed = (clu[:, np.newaxis] + clu[np.newaxis, :]) / 2
        for iparam in range(nparams):
            eta = params[iparam, 0]
            gam = params[iparam, 1]
            b[:, iparam] = fcn_clu_avg(A, Kseed, D, m, eta, gam, modelvar, epsilon)

    elif modeltype == 'clu-diff':
        clu = clustering_coef_bu(A)
        Kseed = np.abs(clu[:, np.newaxis] - clu[np.newaxis, :])
        for iparam in range(nparams):
            eta = params[iparam, 0]
            gam = params[iparam, 1]
            b[:, iparam] = fcn_clu_diff(A, Kseed, D, m, eta, gam, modelvar, epsilon)

    elif modeltype == 'clu-max':
        clu = clustering_coef_bu(A)
        Kseed = np.maximum(clu[:, np.newaxis], clu[np.newaxis, :])
        for iparam in range(nparams):
            eta = params[iparam, 0]
            gam = params[iparam, 1]
            b[:, iparam] = fcn_clu_max(A, Kseed, D, m, eta, gam, modelvar, epsilon)

    elif modeltype == 'clu-min':
        clu = clustering_coef_bu(A)
        Kseed = np.minimum(clu[:, np.newaxis], clu[np.newaxis, :])
        for iparam in range(nparams):
            eta = params[iparam, 0]
            gam = params[iparam, 1]
            b[:, iparam] = fcn_clu_min(A, Kseed, D, m, eta, gam, modelvar, epsilon)

    elif modeltype == 'clu-prod':
        clu = clustering_coef_bu(A)
        Kseed = np.dot(clu, clu.T)
        for iparam in range(nparams):
            eta = params[iparam, 0]
            gam = params[iparam, 1]
            b[:, iparam] = fcn_clu_prod(A, Kseed, D, m, eta, gam, modelvar, epsilon)

    elif modeltype == 'deg-avg':
        kseed = np.sum(A, axis=1)
        Kseed = (kseed[:, np.newaxis] + kseed[np.newaxis, :]) / 2
        for iparam in range(nparams):
            eta = params[iparam, 0]
            gam = params[iparam, 1]
            b[:, iparam] = fcn_deg_avg(A, Kseed, D, m, eta, gam, modelvar, epsilon)

    elif modeltype == 'deg-diff':
        kseed = np.sum(A, axis=1)
        Kseed = np.abs(kseed[:, np.newaxis] - kseed[np.newaxis, :])
        for iparam in range(nparams):
            eta = params[iparam, 0]
            gam = params[iparam, 1]
            b[:, iparam] = fcn_deg_diff(A, Kseed, D, m, eta, gam, modelvar, epsilon)

    elif modeltype == 'deg-max':
        kseed = np.sum(A, axis=1)
        Kseed = np.maximum(kseed[:, np.newaxis], kseed[np.newaxis, :])
        for iparam in range(nparams):
            eta = params[iparam, 0]
            gam = params[iparam, 1]
            b[:, iparam] = fcn_deg_max(A, Kseed, D, m, eta, gam, modelvar, epsilon)

    elif modeltype == 'deg-min':
        kseed = np.sum(A, axis=1)
        Kseed = np.minimum(kseed[:, np.newaxis], kseed[np.newaxis, :])
        for iparam in range(nparams):
            eta = params[iparam, 0]
            gam = params[iparam, 1]
            b[:, iparam] = fcn_deg_min(A, Kseed, D, m, eta, gam, modelvar, epsilon)

    elif modeltype == 'deg-prod':
        kseed = np.sum(A, axis=1)
        Kseed = np.dot(kseed, kseed.T) * (1 - np.eye(n))
        for iparam in range(nparams):
            eta = params[iparam, 0]
            gam = params[iparam, 1]
            b[:, iparam] = fcn_deg_prod(A, Kseed, D, m, eta, gam, modelvar, epsilon)

    elif modeltype == 'neighbors':
        Kseed = np.dot(A, A) * (1 - np.eye(n))
        for iparam in range(nparams):
            eta = params[iparam, 0]
            gam = params[iparam, 1]
            b[:, iparam] = fcn_nghbrs(A, Kseed, D, m, eta, gam, modelvar, epsilon)

    elif modeltype == 'matching':
        Kseed = matching_ind(A)
        Kseed = Kseed + Kseed.T
        for iparam in range(nparams):
            eta = params[iparam, 0]
            gam = params[iparam, 1]
            b[:, iparam] = fcn_matching(A, Kseed, D, m, eta, gam, modelvar, epsilon)

    elif modeltype == 'sptl':
        for iparam in range(nparams):
            eta = params[iparam, 0]
            b[:, iparam] = fcn_sptl(A, D, m, eta, modelvar[0])

    return b


def clustering_coef_bu(A):
    """Placeholder for clustering_coef_bu function.  Replace with your actual implementation."""
    #This is a placeholder. Replace with your actual implementation of clustering_coef_bu
    return np.zeros(A.shape[0])

def matching_ind(A):
    """Placeholder for matching_ind function. Replace with your actual implementation."""
    #This is a placeholder. Replace with your actual implementation of matching_ind
    return np.zeros((A.shape[0], A.shape[1]))


def fcn_clu_avg(A, K, D, m, eta, gam, modelvar, epsilon):
    K = K + epsilon
    n = len(D)
    mseed = np.count_nonzero(A) // 2
    A = A > 0
    mv1 = modelvar[0]
    mv2 = modelvar[1]
    if mv1 == 'powerlaw':
        Fd = D**eta
    elif mv1 == 'exponential':
        Fd = np.exp(eta * D)
    if mv2 == 'powerlaw':
        Fk = K**gam
    elif mv2 == 'exponential':
        Fk = np.exp(gam * K)

    c = clustering_coef_bu(A)
    k = np.sum(A, axis=1)

    Ff = Fd * Fk * (1 - A)
    u, v = np.nonzero(np.triu(np.ones((n, n)), 1))
    indx = (v) * n + u
    P = Ff.flatten()[indx]

    for i in range(mseed + 1, m):
        C = np.concatenate(([0], np.cumsum(P)))
        r = np.sum(np.random.rand() * C[-1] >= C)
        uu = u[r]
        vv = v[r]
        A[uu, vv] = 1
        A[vv, uu] = 1
        k[[uu, vv]] = k[[uu, vv]] + 1
        bu = A[uu, :]
        su = A[bu, :][:, bu]
        bv = A[vv, :]
        sv = A[bv, :][:, bv]
        bth = np.logical_and(bu, bv)
        c[bth] = c[bth] + 2 / (k[bth]**2 - k[bth])
        c[uu] = np.count_nonzero(su) / (k[uu] * (k[uu] - 1))
        c[vv] = np.count_nonzero(sv) / (k[vv] * (k[vv] - 1))
        c[k <= 1] = 0
        bth[[uu, vv]] = True
        K[:, bth] = (c[:, np.newaxis] + c[bth, np.newaxis].T) / 2 + epsilon
        K[bth, :] = (c[:, np.newaxis] + c[bth, np.newaxis].T).T / 2 + epsilon

        if mv2 == 'powerlaw':
            Ff[bth, :] = Fd[bth, :] * (K[bth, :]**gam)
            Ff[:, bth] = Fd[:, bth] * (K[:, bth]**gam)
        elif mv2 == 'exponential':
            Ff[bth, :] = Fd[bth, :] * np.exp(K[bth, :] * gam)
            Ff[:, bth] = Fd[:, bth] * np.exp(K[:, bth] * gam)
        Ff = Ff * (1 - A)
        P = Ff.flatten()[indx]
    b = np.nonzero(np.triu(A, 1))[0]
    return b


def fcn_clu_diff(A, K, D, m, eta, gam, modelvar, epsilon):
    # ... (Implementation similar to fcn_clu_avg, but using np.abs(c[:, np.newaxis] - c[bth, np.newaxis].T)) ...
    K = K + epsilon
    n = len(D)
    mseed = np.count_nonzero(A) // 2
    A = A > 0
    mv1 = modelvar[0]
    mv2 = modelvar[1]
    if mv1 == 'powerlaw':
        Fd = D**eta
    elif mv1 == 'exponential':
        Fd = np.exp(eta * D)
    if mv2 == 'powerlaw':
        Fk = K**gam
    elif mv2 == 'exponential':
        Fk = np.exp(gam * K)

    c = clustering_coef_bu(A)
    k = np.sum(A, axis=1)

    Ff = Fd * Fk * (1 - A)
    u, v = np.nonzero(np.triu(np.ones((n, n)), 1))
    indx = (v) * n + u
    P = Ff.flatten()[indx]

    for i in range(mseed + 1, m):
        C = np.concatenate(([0], np.cumsum(P)))
        r = np.sum(np.random.rand() * C[-1] >= C)
        uu = u[r]
        vv = v[r]
        A[uu, vv] = 1
        A[vv, uu] = 1
        k[[uu, vv]] = k[[uu, vv]] + 1
        bu = A[uu, :]
        su = A[bu, :][:, bu]
        bv = A[vv, :]
        sv = A[bv, :][:, bv]
        bth = np.logical_and(bu, bv)
        c[bth] = c[bth] + 2 / (k[bth]**2 - k[bth])
        c[uu] = np.count_nonzero(su) / (k[uu] * (k[uu] - 1))
        c[vv] = np.count_nonzero(sv) / (k[vv] * (k[vv] - 1))
        c[k <= 1] = 0
        bth[[uu, vv]] = True
        K[:, bth] = np.abs(c[:, np.newaxis] - c[bth, np.newaxis].T) + epsilon
        K[bth, :] = np.abs(c[:, np.newaxis] - c[bth, np.newaxis].T).T + epsilon

        if mv2 == 'powerlaw':
            Ff[bth, :] = Fd[bth, :] * (K[bth, :]**gam)
            Ff[:, bth] = Fd[:, bth] * (K[:, bth]**gam)
        elif mv2 == 'exponential':
            Ff[bth, :] = Fd[bth, :] * np.exp(K[bth, :] * gam)
            Ff[:, bth] = Fd[:, bth] * np.exp(K[:, bth] * gam)
        Ff = Ff * (1 - A)
        P = Ff.flatten()[indx]
    b = np.nonzero(np.triu(A, 1))[0]
    return b

def fcn_clu_max(A, K, D, m, eta, gam, modelvar, epsilon):
    # ... (Implementation similar to fcn_clu_avg, but using np.maximum) ...
    K = K + epsilon
    n = len(D)
    mseed = np.count_nonzero(A) // 2
    A = A > 0
    mv1 = modelvar[0]
    mv2 = modelvar[1]
    if mv1 == 'powerlaw':
        Fd = D**eta
    elif mv1 == 'exponential':
        Fd = np.exp(eta * D)
    if mv2 == 'powerlaw':
        Fk = K**gam
    elif mv2 == 'exponential':
        Fk = np.exp(gam * K)

    c = clustering_coef_bu(A)
    k = np.sum(A, axis=1)

    Ff = Fd * Fk * (1 - A)
    u, v = np.nonzero(np.triu(np.ones((n, n)), 1))
    indx = (v) * n + u
    P = Ff.flatten()[indx]

    for i in range(mseed + 1, m):
        C = np.concatenate(([0], np.cumsum(P)))
        r = np.sum(np.random.rand() * C[-1] >= C)
        uu = u[r]
        vv = v[r]
        A[uu, vv] = 1
        A[vv, uu] = 1
        k[[uu, vv]] = k[[uu, vv]] + 1
        bu = A[uu, :]
        su = A[bu, :][:, bu]
        bv = A[vv, :]
        sv = A[bv, :][:, bv]
        bth = np.logical_and(bu, bv)
        c[bth] = c[bth] + 2 / (k[bth]**2 - k[bth])
        c[uu] = np.count_nonzero(su) / (k[uu] * (k[uu] - 1))
        c[vv] = np.count_nonzero(sv) / (k[vv] * (k[vv] - 1))
        c[k <= 1] = 0
        bth[[uu, vv]] = True
        K[:, bth] = np.maximum(c[:, np.newaxis], c[bth, np.newaxis].T) + epsilon
        K[bth, :] = np.maximum(c[:, np.newaxis], c[bth, np.newaxis].T).T + epsilon

        if mv2 == 'powerlaw':
            Ff[bth, :] = Fd[bth, :] * (K[bth, :]**gam)
            Ff[:, bth] = Fd[:, bth] * (K[:, bth]**gam)
        elif mv2 == 'exponential':
            Ff[bth, :] = Fd[bth, :] * np.exp(K[bth, :] * gam)
            Ff[:, bth] = Fd[:, bth] * np.exp(K[:, bth] * gam)
        Ff = Ff * (1 - A)
        P = Ff.flatten()[indx]
    b = np.nonzero(np.triu(A, 1))[0]
    return b

def fcn_clu_min(A, K, D, m, eta, gam, modelvar, epsilon):
    # ... (Implementation similar to fcn_clu_avg, but using np.minimum) ...
    K = K + epsilon
    n = len(D)
    mseed = np.count_nonzero(A) // 2
    A = A > 0
    mv1 = modelvar[0]
    mv2 = modelvar[1]
    if mv1 == 'powerlaw':
        Fd = D**eta
    elif mv1 == 'exponential':
        Fd = np.exp(eta * D)
    if mv2 == 'powerlaw':
        Fk = K**gam
    elif mv2 == 'exponential':
        Fk = np.exp(gam * K)

    c = clustering_coef_bu(A)
    k = np.sum(A, axis=1)

    Ff = Fd * Fk * (1 - A)
    u, v = np.nonzero(np.triu(np.ones((n, n)), 1))
    indx = (v) * n + u
    P = Ff.flatten()[indx]

    for i in range(mseed + 1, m):
        C = np.concatenate(([0], np.cumsum(P)))
        r = np.sum(np.random.rand() * C[-1] >= C)
        uu = u[r]
        vv = v[r]
        A[uu, vv] = 1
        A[vv, uu] = 1
        k[[uu, vv]] = k[[uu, vv]] + 1
        bu = A[uu, :]
        su = A[bu, :][:, bu]
        bv = A[vv, :]
        sv = A[bv, :][:, bv]
        bth = np.logical_and(bu, bv)
        c[bth] = c[bth] + 2 / (k[bth]**2 - k[bth])
        c[uu] = np.count_nonzero(su) / (k[uu] * (k[uu] - 1))
        c[vv] = np.count_nonzero(sv) / (k[vv] * (k[vv] - 1))
        c[k <= 1] = 0
        bth[[uu, vv]] = True
        K[:, bth] = np.minimum(c[:, np.newaxis], c[bth, np.newaxis].T) + epsilon
        K[bth, :] = np.minimum(c[:, np.newaxis], c[bth, np.newaxis].T).T + epsilon

        if mv2 == 'powerlaw':
            Ff[bth, :] = Fd[bth, :] * (K[bth, :]**gam)
            Ff[:, bth] = Fd[:, bth] * (K[:, bth]**gam)
        elif mv2 == 'exponential':
            Ff[bth, :] = Fd[bth, :] * np.exp(K[bth, :] * gam)
            Ff[:, bth] = Fd[:, bth] * np.exp(K[:, bth] * gam)
        Ff = Ff * (1 - A)
        P = Ff.flatten()[indx]
    b = np.nonzero(np.triu(A, 1))[0]
    return b


def fcn_clu_prod(A, K, D, m, eta, gam, modelvar, epsilon):
    # ... (Implementation similar to fcn_clu_avg, but using np.dot(c,c.T)) ...
    K = K + epsilon
    n = len(D)
    mseed = np.count_nonzero(A) // 2
    A = A > 0
    mv1 = modelvar[0]
    mv2 = modelvar[1]
    if mv1 == 'powerlaw':
        Fd = D**eta
    elif mv1 == 'exponential':
        Fd = np.exp(eta * D)
    if mv2 == 'powerlaw':
        Fk = K**gam
    elif mv2 == 'exponential':
        Fk = np.exp(gam * K)

    c = clustering_coef_bu(A)
    k = np.sum(A, axis=1)

    Ff = Fd * Fk * (1 - A)
    u, v = np.nonzero(np.triu(np.ones((n, n)), 1))
    indx = (v) * n + u
    P = Ff.flatten()[indx]

    for i in range(mseed + 1, m):
        C = np.concatenate(([0], np.cumsum(P)))
        r = np.sum(np.random.rand() * C[-1] >= C)
        uu = u[r]
        vv = v[r]
        A[uu, vv] = 1
        A[vv, uu] = 1
        k[[uu, vv]] = k[[uu, vv]] + 1
        bu = A[uu, :]
        su = A[bu, :][:, bu]
        bv = A[vv, :]
        sv = A[bv, :][:, bv]
        bth = np.logical_and(bu, bv)
        c[bth] = c[bth] + 2 / (k[bth]**2 - k[bth])
        c[uu] = np.count_nonzero(su) / (k[uu] * (k[uu] - 1))
        c[vv] = np.count_nonzero(sv) / (k[vv] * (k[vv] - 1))
        c[k <= 1] = 0
        bth[[uu, vv]] = True
        K[bth, :] = np.dot(c[bth, :], c.T) + epsilon
        K[:, bth] = np.dot(c, c[bth, :].T) + epsilon

        if mv2 == 'powerlaw':
            Ff[bth, :] = Fd[bth, :] * (K[bth, :]**gam)
            Ff[:, bth] = Fd[:, bth] * (K[:, bth]**gam)
        elif mv2 == 'exponential':
            Ff[bth, :] = Fd[bth, :] * np.exp(K[bth, :] * gam)
            Ff[:, bth] = Fd[:, bth] * np.exp(K[:, bth] * gam)
        Ff = Ff * (1 - A)
        P = Ff.flatten()[indx]
    b = np.nonzero(np.triu(A, 1))[0]
    return b


def fcn_deg_avg(A, K, D, m, eta, gam, modelvar, epsilon):
    n = len(D)
    mseed = np.count_nonzero(A) // 2
    k = np.sum(A, axis=1)
    u, v = np.nonzero(np.triu(np.ones((n, n)), 1))
    indx = (v) * n + u
    D = D.flatten()[indx]
    mv1 = modelvar[0]
    mv2 = modelvar[1]
    if mv1 == 'powerlaw':
        Fd = D**eta
    elif mv1 == 'exponential':
        Fd = np.exp(eta * D)
    K = K + epsilon
    if mv2 == 'powerlaw':
        Fk = K**gam
    elif mv2 == 'exponential':
        Fk = np.exp(gam * K)
    P = Fd * Fk.flatten()[indx] * (1 - A.flatten()[indx])
    b = np.zeros(m)
    b[:mseed] = np.nonzero(A.flatten()[indx])[0]
    for i in range(mseed, m):
        C = np.concatenate(([0], np.cumsum(P)))
        r = np.sum(np.random.rand() * C[-1] >= C)
        w = [u[r], v[r]]
        k[w] = k[w] + 1
        if mv2 == 'powerlaw':
            Fk[:, w] = ((k + k[w[0]]) / 2 + epsilon)**gam
            Fk[w, :] = ((k + k[w[1]]) / 2 + epsilon)**gam
        elif mv2 == 'exponential':
            Fk[:, w] = np.exp(((k + k[w[0]]) / 2 + epsilon) * gam)
            Fk[w, :] = np.exp(((k + k[w[1]]) / 2 + epsilon) * gam)
        P = Fd * Fk.flatten()[indx]
        b[i] = r
        P[b[:i+1].astype(int)] = 0
    b = indx[b.astype(int)]
    return b

def fcn_deg_diff(A, K, D, m, eta, gam, modelvar, epsilon):
    # ... (Implementation similar to fcn_deg_avg, but using np.abs(k - k[w])) ...
    n = len(D)
    mseed = np.count_nonzero(A) // 2
    k = np.sum(A, axis=1)
    u, v = np.nonzero(np.triu(np.ones((n, n)), 1))
    indx = (v) * n + u
    D = D.flatten()[indx]
    mv1 = modelvar[0]
    mv2 = modelvar[1]
    if mv1 == 'powerlaw':
        Fd = D**eta
    elif mv1 == 'exponential':
        Fd = np.exp(eta * D)
    K = K + epsilon
    if mv2 == 'powerlaw':
        Fk = K**gam
    elif mv2 == 'exponential':
        Fk = np.exp(gam * K)
    P = Fd * Fk.flatten()[indx] * (1 - A.flatten()[indx])
    b = np.zeros(m)
    b[:mseed] = np.nonzero(A.flatten()[indx])[0]
    for i in range(mseed, m):
        C = np.concatenate(([0], np.cumsum(P)))
        r = np.sum(np.random.rand() * C[-1] >= C)
        w = [u[r], v[r]]
        k[w] = k[w] + 1
        if mv2 == 'powerlaw':
            Fk[:, w] = (np.abs(k - k[w[0]]) + epsilon)**gam
            Fk[w, :] = (np.abs(k - k[w[1]]) + epsilon)**gam
        elif mv2 == 'exponential':
            Fk[:, w] = np.exp((np.abs(k - k[w[0]]) + epsilon) * gam)
            Fk[w, :] = np.exp((np.abs(k - k[w[1]]) + epsilon) * gam)
        P = Fd * Fk.flatten()[indx]
        b[i] = r
        P[b[:i+1].astype(int)] = 0
    b = indx[b.astype(int)]
    return b

def fcn_deg_min(A, K, D, m, eta, gam, modelvar, epsilon):
    # ... (Implementation similar to fcn_deg_avg, but using np.minimum) ...
    n = len(D)
    mseed = np.count_nonzero(A) // 2
    k = np.sum(A, axis=1)
    u, v = np.nonzero(np.triu(np.ones((n, n)), 1))
    indx = (v) * n + u
    D = D.flatten()[indx]
    mv1 = modelvar[0]
    mv2 = modelvar[1]
    if mv1 == 'powerlaw':
        Fd = D**eta
    elif mv1 == 'exponential':
        Fd = np.exp(eta * D)
    K = K + epsilon
    if mv2 == 'powerlaw':
        Fk = K**gam
    elif mv2 == 'exponential':
        Fk = np.exp(gam * K)
    P = Fd * Fk.flatten()[indx] * (1 - A.flatten()[indx])
    b = np.zeros(m)
    b[:mseed] = np.nonzero(A.flatten()[indx])[0]
    for i in range(mseed, m):
        C = np.concatenate(([0], np.cumsum(P)))
        r = np.sum(np.random.rand() * C[-1] >= C)
        w = [u[r], v[r]]
        k[w] = k[w] + 1
        if mv2 == 'powerlaw':
            Fk[:, w] = (np.minimum(k, k[w[0]]) + epsilon)**gam
            Fk[w, :] = (np.minimum(k, k[w[1]]) + epsilon)**gam
        elif mv2 == 'exponential':
            Fk[:, w] = np.exp((np.minimum(k, k[w[0]]) + epsilon) * gam)
            Fk[w, :] = np.exp((np.minimum(k, k[w[1]]) + epsilon) * gam)
        P = Fd * Fk.flatten()[indx]
        b[i] = r
        P[b[:i+1].astype(int)] = 0
    b = indx[b.astype(int)]
    return b

def fcn_deg_max(A, K, D, m, eta, gam, modelvar, epsilon):
    # ... (Implementation similar to fcn_deg_avg, but using np.maximum) ...
    n = len(D)
    mseed = np.count_nonzero(A) // 2
    k = np.sum(A, axis=1)
    u, v = np.nonzero(np.triu(np.ones((n, n)), 1))
    indx = (v) * n + u
    D = D.flatten()[indx]
    mv1 = modelvar[0]
    mv2 = modelvar[1]
    if mv1 == 'powerlaw':
        Fd = D**eta
    elif mv1 == 'exponential':
        Fd = np.exp(eta * D)
    K = K + epsilon
    if mv2 == 'powerlaw':
        Fk = K**gam
    elif mv2 == 'exponential':
        Fk = np.exp(gam * K)
    P = Fd * Fk.flatten()[indx] * (1 - A.flatten()[indx])
    b = np.zeros(m)
    b[:mseed] = np.nonzero(A.flatten()[indx])[0]
    for i in range(mseed, m):
        C = np.concatenate(([0], np.cumsum(P)))
        r = np.sum(np.random.rand() * C[-1] >= C)
        w = [u[r], v[r]]
        k[w] = k[w] + 1
        if mv2 == 'powerlaw':
            Fk[:, w] = (np.maximum(k, k[w[0]]) + epsilon)**gam
            Fk[w, :] = (np.maximum(k, k[w[1]]) + epsilon)**gam
        elif mv2 == 'exponential':
            Fk[:, w] = np.exp((np.maximum(k, k[w[0]]) + epsilon) * gam)
            Fk[w, :] = np.exp((np.maximum(k, k[w[1]]) + epsilon) * gam)
        P = Fd * Fk.flatten()[indx]
        b[i] = r
        P[b[:i+1].astype(int)] = 0
    b = indx[b.astype(
