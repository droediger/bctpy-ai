# Translated from erange.m

import numpy as np

def erange(CIJ):
    #ERANGE     Shortcuts
    #
    #   [Erange,eta,Eshort,fs] = erange(CIJ);
    #
    #   Shorcuts are central edges which significantly reduce the
    #   characteristic path length in the network.
    #
    #   Input:      CIJ,        binary directed connection matrix
    #
    #   Outputs:    Erange,     range for each edge, i.e. the length of the
    #                           shortest path from i to j for edge c(i,j) AFTER
    #                           the edge has been removed from the graph.
    #               eta         average range for entire graph.
    #               Eshort      entries are ones for shortcut edges.
    #               fs          fraction of shortcuts in the graph.
    #
    #   Follows the treatment of 'shortcuts' by Duncan Watts
    #
    #
    #   Olaf Sporns, Indiana University, 2002/2007/2008

    N = CIJ.shape[0]  # Number of nodes
    K = np.count_nonzero(CIJ)  # Number of edges
    Erange = np.zeros((N, N))  # Initialize range matrix
    i, j = np.nonzero(CIJ == 1)  # Find indices of edges

    def reachdist(CIJcut): # Assumed to be defined elsewhere.  Replace with your actual implementation.
        # Placeholder for reachdist function.  This should calculate shortest paths.
        # Replace this with your actual reachdist implementation.  
        # This is crucial for correct results.
        pass

    for c in range(len(i)):
        CIJcut = np.copy(CIJ)  # Create a copy to avoid modifying the original matrix
        CIJcut[i[c], j[c]] = 0  # Remove the current edge
        _, D = reachdist(CIJcut)  # Calculate shortest path distances
        Erange[i[c], j[c]] = D[i[c], j[c]]  # Store the range for the current edge

    # average range (ignore Inf)
    eta = np.sum(Erange[(Erange > 0) & (Erange < np.inf)]) / np.count_nonzero((Erange > 0) & (Erange < np.inf))

    # Original entries of D are ones, thus entries of Erange
    # must be two or greater.
    # If Erange(i,j) > 2, then the edge is a shortcut.
    # 'fshort' is the fraction of shortcuts over the entire graph.

    Eshort = Erange > 2  # Identify shortcut edges
    fs = np.count_nonzero(Eshort) / K  # Calculate the fraction of shortcuts

    return Erange, eta, Eshort, fs

