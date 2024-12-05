# Translated from retrieve_shortest_path.m

import numpy as np

def retrieve_shortest_path(s, t, hops, Pmat):
    # RETRIEVE_SHORTEST_PATH        Retrieval of shortest path
    #
    #   This function finds the sequence of nodes that comprise the shortest
    #   path between a given source and target node.
    #
    #   Inputs:
    #       s,
    #           Source node: i.e. node where the shortest path begins. 
    #    	t,
    #           Target node: i.e. node where the shortest path ends.
    #       hops,
    #           Number of edges in the path. This matrix may be obtained as the
    #           second output argument of the function "distance_wei_floyd.m".
    #       Pmat,
    #           Pmat is a matrix whose elements {k,t} indicate the next node in
    #           the shortest path between k and t. This matrix may be obtained
    #           as the third output of the function "distance_wei_floyd.m"
    #
    #   Output:
    #       path,
    #           Nodes comprising the shortest path between nodes s and t.
    #
    #
    #   Andrea Avena-Koenigsberger and Joaquin Goñi, IU, 2012

    path_length = hops[s-1,t-1] # Adjust for 0-based indexing in Python
    if path_length != 0:
        path = np.empty(path_length+1, dtype=int) #Initialize as numpy array of integers
        path[0] = s
        for ind in range(1,path_length+1):
            s = Pmat[s-1,t-1] +1 # Adjust for 0-based indexing and add 1 to match MATLAB indexing
            path[ind] = s
        return path
    else:
        return np.array([]) #Return an empty numpy array



