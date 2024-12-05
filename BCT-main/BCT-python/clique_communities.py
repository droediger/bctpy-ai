# Translated from clique_communities.m

import numpy as np

def clique_communities(A, cq_thr):
    # CLIQUE_COMMUNITIES     Overlapping community structure via clique percolation
    #
    #   M = clique_communities(A, cq_thr)
    #
    #   The optimal community structure is a subdivision of the network into
    #   groups of nodes which have a high number of within-group connections
    #   and a low number of between group connections.
    #
    #   This algorithm uncovers overlapping community structure in binary
    #   undirected networks via the clique percolation method.
    #
    #   Inputs:
    #       A,          Binary undirected connection matrix.
    #
    #      	cq_thr,     Clique size threshold (integer). Larger clique size
    #                   thresholds potentially result in larger communities.
    #
    #   Output:     
    #       M,          Overlapping community-affiliation matrix
    #                   Binary matrix of size CxN [communities x nodes]
    #
    #   Algorithms:
    #       Bron–Kerbosch algorithm for detection of maximal cliques.
    #       Dulmage-Mendelsohn decomposition for detection of components
    #                   (implemented in get_components.m)
    #
    #
    #   Note: This algorithm can be slow and memory intensive in large
    #   matrices. The algorithm requires the function get_components.m
    #
    #   Reference: Palla et al. (2005) Nature 435, 814-818.
    #
    #   Mika Rubinov, Janelia HHMI, 2017

    if not np.allclose(A, A.T):
        raise ValueError('A must be undirected.')
    if A.shape[0] != A.shape[1]:
        raise ValueError('A must be square.')
    if not isinstance(A, np.ndarray) or not A.dtype == bool:
        A = A.astype(bool)
    
    n = A.shape[0]                                 # number of nodes
    np.fill_diagonal(A, 0)                        # clear diagonal
    MQ = maximal_cliques(A, n)                     # get maximal cliques
    Cq = np.array(MQ).T                           # convert to matrix
    Cq = Cq[np.sum(Cq, axis=0) >= cq_thr, :]       # remove subthreshold cliques
    Ov = Cq @ Cq.T                                 # compute clique overlap
    Ov_thr = (Ov >= cq_thr - 1).astype(int)        # keep percolating cliques

    Cq_components = get_components(Ov_thr)         # find components 

    m = np.max(Cq_components)                     # get number of components
    M = np.zeros((m, n), dtype=int)                # collect communities
    for i in range(m):
        M[i, np.any(Cq[Cq_components == i + 1, :], axis=0)] = 1
    return M


def maximal_cliques(A, n):             # Bron-Kerbosch algorithm
    MQ = [[] for _ in range(1000 * n)]

    R = np.zeros(n, dtype=bool)               #current
    P = np.ones(n, dtype=bool)                #prospective
    X = np.zeros(n, dtype=bool)               #processed
    q = 0

    BK(R, P, X)

    def BK(R, P, X):
        if not np.any(P | X):
            nonlocal q
            q += 1
            MQ[q-1] = np.nonzero(R)[0].tolist()
        else:
            U_p = np.nonzero(np.any([P, X], axis=0))[0]
            
            # Simulate MATLAB's max with tie-breaking based on index
            scores = A[:,U_p].T @ P.astype(int)
            idx = np.argmax(scores)
            u_p = U_p[idx]
            
            U = np.nonzero(np.all([P, ~A[:,u_p]], axis=0))[0]
            for u in U:
                Nu = A[:,u]
                P[u] = False
                Rnew = np.copy(R)
                Rnew[u] = True
                Pnew = np.all([P, Nu], axis=0)
                Xnew = np.all([X, Nu], axis=0)
                BK(Rnew, Pnew, Xnew)
                X[u] = True
    return MQ[:q]



def get_components(A):
    # Placeholder for get_components.m functionality.  Replace with actual implementation if available.
    # This is a simplified example and may not accurately reflect the original function's behavior.
    n = A.shape[0]
    visited = np.zeros(n,dtype=bool)
    components = np.zeros(n,dtype=int)
    component_index = 0
    for i in range(n):
        if not visited[i]:
            component_index += 1
            stack = [i]
            while stack:
                j = stack.pop()
                if not visited[j]:
                    visited[j] = True
                    components[j] = component_index
                    neighbors = np.nonzero(A[j])[0]
                    stack.extend(neighbors)
    return components

