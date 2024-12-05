# Translated from make_motif34lib.m

import numpy as np

def motif3generate():
    """Generates 3-node motifs."""
    n = 0
    M = np.zeros((54, 6), dtype=bool)  # Isomorphs
    CL = np.zeros((54, 6), dtype=np.uint8)  # Canonical labels (predecessors of IDs)
    cl = np.zeros((1, 6), dtype=np.uint8)
    for i in range(2**6):  # Loop through all subgraphs
        m = bin(i)[2:].zfill(6)
        G = np.array([
            [0, int(m[2]), int(m[4])],
            [int(m[0]), 0, int(m[5])],
            [int(m[1]), int(m[3]), 0]
        ])
        Ko = np.sum(G, axis=1)
        Ki = np.sum(G, axis=0)
        if np.all(Ko | Ki):  # If subgraph weakly-connected
            n += 1
            cl[:] = np.sort(np.concatenate((Ko, Ki))).reshape(1,-1)
            CL[n - 1, :] = cl  # Assign motif label to isomorph
            M[n - 1, :] = G.flatten()[1:7]
    u1, u2, ID = np.unique(CL, axis=0, return_index=True, return_inverse=True)  # Convert CLs into motif IDs

    # Convert IDs into Sporns & Kotter classification
    id_mika = np.array([1, 3, 4, 6, 7, 8, 11])
    id_olaf = -np.array([3, 6, 1, 11, 4, 7, 8])
    for id in range(len(id_mika)):
        ID[ID == id_mika[id]] = id_olaf[id]
    ID = np.abs(ID)

    X, ind = np.sort(ID, axis=None, kind='mergesort'), np.argsort(ID, axis=None, kind='mergesort')
    ID = ID[ind]  # Sort IDs
    M = M[ind, :]  # Sort isomorphs
    N = np.sum(M, axis=1)  # Number of edges
    Mn = np.sum(M * (10**np.arange(5, -1, -1)), axis=1).astype(np.uint32)  # M as a single number
    return M, Mn, ID, N


def motif4generate():
    """Generates 4-node motifs."""
    n = 0
    M = np.zeros((3834, 12), dtype=bool)  # Isomorphs
    CL = np.zeros((3834, 16), dtype=np.uint8)  # Canonical labels (predecessors of IDs)
    cl = np.zeros((1, 16), dtype=np.uint8)
    for i in range(2**12):  # Loop through all subgraphs
        m = bin(i)[2:].zfill(12)
        G = np.array([
            [0, int(m[3]), int(m[6]), int(m[9])],
            [int(m[0]), 0, int(m[7]), int(m[10])],
            [int(m[1]), int(m[4]), 0, int(m[11])],
            [int(m[2]), int(m[5]), int(m[8]), 0]
        ])
        Gs = G + G.T
        v = Gs[0, :]
        for j in range(2):
            v = np.any(Gs[v != 0, :], axis=0) + v
        if np.any(v):  # If subgraph weakly connected
            n += 1
            G2 = (G @ G) != 0
            Ko = np.sum(G, axis=1)
            Ki = np.sum(G, axis=0)
            Ko2 = np.sum(G2, axis=1)
            Ki2 = np.sum(G2, axis=0)
            cl[:] = np.sort(np.concatenate((Ki, Ko, Ki2, Ko2))).reshape(1,-1)
            CL[n - 1, :] = cl  # Assign motif label to isomorph
            M[n - 1, :] = G.flatten()[1:13]
    u1, u2, ID = np.unique(CL, axis=0, return_index=True, return_inverse=True)  # Convert CLs into motif IDs
    X, ind = np.sort(ID, axis=None, kind='mergesort'), np.argsort(ID, axis=None, kind='mergesort')
    ID = ID[ind]  # Sort IDs
    M = M[ind, :]  # Sort isomorphs
    N = np.sum(M, axis=1)  # Number of edges
    Mn = np.sum(M * (10**np.arange(11, -1, -1)), axis=1).astype(np.uint64)  # M as a single number
    return M, Mn, ID, N


def make_motif34lib():
    """Generates the motif34lib.mat library."""
    M3, M3n, ID3, N3 = motif3generate()
    M4, M4n, ID4, N4 = motif4generate()
    np.savez_compressed('motif34lib', M3=M3, M3n=M3n, ID3=ID3, N3=N3, M4=M4, M4n=M4n, ID4=ID4, N4=N4)


make_motif34lib()

