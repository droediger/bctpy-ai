# Translated from agreement.m

import numpy as np

def agreement(ci, buffsz=1000):
    """
    Agreement matrix from clusters

    Args:
        ci: A set of vertex partitions of dimensions [vertex x partition]. 
            Each column in ci contains the assignments of each vertex to a class/community/module.
        buffsz: Optional second argument to set buffer size for processing large datasets.

    Returns:
        D: Agreement matrix.  A square [vertex x vertex] matrix whose elements indicate the number of times any two vertices were assigned to the same class.
    """

    n = ci.shape[1]  # Number of partitions

    if n <= buffsz:
        # For smaller datasets, compute the agreement matrix directly.
        ind = dummyvar(ci) #Assumed to be defined elsewhere
        D = np.dot(ind, ind.T)
    else:
        # For larger datasets, compute the agreement matrix in pieces to manage memory.
        a = np.arange(1, n + 1, buffsz)
        b = np.arange(buffsz, n + 1, buffsz)
        if len(a) != len(b):
            b = np.append(b, n)
        x = np.column_stack((a, b))
        nbuff = x.shape[0]

        D = np.zeros((ci.shape[0], ci.shape[0]))
        for i in range(nbuff):
            y = ci[:, x[i, 0]-1:x[i, 1]]
            ind = dummyvar(y) #Assumed to be defined elsewhere
            D += np.dot(ind, ind.T)

    # Set diagonal to zero to exclude self-agreement.
    np.fill_diagonal(D, 0)
    return D

def dummyvar(ci):
    #Dummy function to replace MATLAB's dummyvar.  Needs to be properly defined based on the original dummyvar function.
    #This is a placeholder, adapt this based on your dummyvar implementation.
    n_vertices = ci.shape[0]
    n_partitions = ci.shape[1]
    ind = np.zeros((n_vertices, np.max(ci)+1))
    for i in range(n_partitions):
        ind[np.arange(n_vertices),ci[:,i]] += 1

    return ind


