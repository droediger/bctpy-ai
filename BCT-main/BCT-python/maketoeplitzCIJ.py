# Translated from maketoeplitzCIJ.m

import numpy as np

def maketoeplitzCIJ(N, K, s):
    # MAKETOEPLITZCIJ    A synthetic directed network with Gaussian drop-off of
    #                   connectivity with distance
    #
    #   CIJ = maketoeprandCIJ(N,K,s)
    #
    #   This function generates a directed network with a Gaussian drop-off in
    #   edge density with increasing distance from the main diagonal. There are
    #   toroidal boundary conditions (i.e. no ring-like "wrapping around").
    #
    #   Inputs:     N,      number of vertices
    #               K,      number of edges
    #               s,      standard deviation of toeplitz
    #
    #   Output:     CIJ,    connection matrix
    #
    #   Note: no connections are placed on the main diagonal.
    #
    #
    # Olaf Sporns, Indiana University, 2005/2007

    profile = np.exp(-(np.arange(1, N) - 0.5)**2 / (2 * s**2)) / (s * np.sqrt(2 * np.pi)) #Gaussian profile
    template = np.zeros((N,N))
    template[0,1:] = profile
    template[1:,0] = profile
    template = np.tile(profile,(N,1)) + np.tile(profile.reshape(N,1),(1,N)) - np.diag(np.diag(np.tile(profile,(N,1)) + np.tile(profile.reshape(N,1),(1,N))))
    template = template * (K / np.sum(template))
    CIJ = np.zeros((N, N))

    while np.sum(CIJ) != K:
        CIJ = np.random.rand(N, N) < template

    return CIJ


