# Translated from breadthdist.m

import numpy as np

def breadthdist(CIJ):
    # BREADTHDIST      Reachability and distance matrices
    #
    #   [R,D] = breadthdist(CIJ);
    #
    #   The binary reachability matrix describes reachability between all pairs
    #   of nodes. An entry (u,v)=1 means that there exists a path from node u
    #   to node v; alternatively (u,v)=0.
    #
    #   The distance matrix contains lengths of shortest paths between all
    #   pairs of nodes. An entry (u,v) represents the length of shortest path 
    #   from node u to  node v. The average shortest path length is the 
    #   characteristic path length of the network.
    #
    #   Input:      CIJ,     binary (directed/undirected) connection matrix
    #
    #   Outputs:    R,       reachability matrix
    #               D,       distance matrix
    #
    #   Note: slower but less memory intensive than "reachdist.m".
    #
    #   Algorithm: Breadth-first search.
    #
    #
    #   Olaf Sporns, Indiana University, 2002/2007/2008

    N = CIJ.shape[0] #Get the number of nodes

    D = np.zeros((N,N)) # Initialize distance matrix
    for i in range(N):
        D[i,:] = breadth(CIJ,i+1) #Apply breadth-first search to each node

    # replace zeros with 'Inf's
    D[D==0] = np.inf

    # construct R
    R = (D!=np.inf).astype(int) #Reachability matrix


    return R, D


def breadth(CIJ,i):
    # Assumed to be defined elsewhere, implementing the breadth-first search algorithm.  
    # This function should take the connectivity matrix CIJ and a starting node i as input,
    # and return a 1D numpy array representing the shortest path distances from node i to all other nodes.
    #  Replace this with your actual breadth-first search implementation.
    N = CIJ.shape[0]
    dist = np.zeros(N)
    queue = [i]
    visited = [False] * N
    visited[i-1] = True
    dist[i-1] = 0
    while queue:
        u = queue.pop(0)
        for v in range(N):
            if CIJ[u-1,v] == 1 and not visited[v]:
                visited[v] = True
                dist[v] = dist[u-1] + 1
                queue.append(v+1)
    return dist

