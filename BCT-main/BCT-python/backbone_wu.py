# Translated from backbone_wu.m

import numpy as np

def backbone_wu(CIJ, avgdeg):
    """
    Computes the backbone of a weighted, undirected connection matrix using a minimum-spanning-tree based algorithm.

    Args:
        CIJ (numpy.ndarray): Connection/adjacency matrix (weighted, undirected).
        avgdeg (float): Desired average degree of the backbone.

    Returns:
        tuple: A tuple containing:
            - CIJtree (numpy.ndarray): Connection matrix of the minimum spanning tree of CIJ.
            - CIJclus (numpy.ndarray): Connection matrix of the minimum spanning tree plus strongest connections up to an average degree 'avgdeg'.
    """
    N = CIJ.shape[0]
    CIJtree = np.zeros((N, N))

    # Find strongest edge (note if multiple edges are tied, only use the first one)
    i, j = np.unravel_index(np.argmax(CIJ), CIJ.shape)
    im = np.array([i, i])
    jm = np.array([j, j])

    # Copy into tree graph
    CIJtree[im[0], jm[0]] = CIJ[im[0], jm[0]]
    CIJtree[jm[0], im[0]] = CIJ[jm[0], im[0]]
    in_nodes = np.array([im[0],jm[0]])
    out_nodes = np.setdiff1d(np.arange(N), in_nodes)

    # Repeat N-2 times
    for n in range(N - 2):
        # Find strongest link between 'in' and 'out', ignore tied ranks
        sub_CIJ = CIJ[in_nodes[:,None], out_nodes]
        i, j = np.unravel_index(np.argmax(sub_CIJ), sub_CIJ.shape)
        im = in_nodes[i]
        jm = out_nodes[j]

        # Copy into tree graph
        CIJtree[im, jm] = CIJ[im, jm]
        CIJtree[jm, im] = CIJ[jm, im]
        in_nodes = np.append(in_nodes, jm)
        out_nodes = np.setdiff1d(np.arange(N), in_nodes)

    # Now add connections back, with the total number of added connections determined by the desired 'avgdeg'
    CIJnotintree = CIJ * (1 - CIJtree)
    a = np.sort(CIJnotintree[CIJnotintree != 0])[::-1]
    cutoff = int(avgdeg * N - 2 * (N - 1))
    if cutoff < 0:
        cutoff = 0
    elif cutoff >= len(a):
        cutoff = len(a) -1

    thr = a[cutoff] if len(a) > cutoff else 0

    CIJclus = CIJtree + (CIJnotintree * (CIJnotintree >= thr))
    return CIJtree, CIJclus

