# Translated from diffusion_efficiency.m

import numpy as np

def diffusion_efficiency(adj):
    # DIFFUSION_EFFICIENCY      Global mean and pair-wise diffusion efficiency
    #
    #   [GEdiff,Ediff] = diffusion_efficiency(adj);
    #
    #   The diffusion efficiency between nodes i and j is the inverse of the
    #   mean first passage time from i to j, that is the expected number of
    #   steps it takes a random walker starting at node i to arrive for the
    #   first time at node j. Note that the mean first passage time is not a
    #   symmetric measure -- mfpt(i,j) may be different from mfpt(j,i) -- and
    #   the pair-wise diffusion efficiency matrix is hence also not symmetric.
    #
    #
    #   Input:
    #       adj,    Weighted/Unweighted, directed/undirected adjacency matrix
    #
    #
    #   Outputs:
    #       GEdiff, Mean Global diffusion efficiency (scalar)
    #       Ediff,  Pair-wise diffusion efficiency (matrix)
    #
    #
    #   References: Goñi J, et al (2013) PLoS ONE
    #
    #   Joaquin Goñi and Andrea Avena-Koenigsberger, IU Bloomington, 2012

    n = adj.shape[0] #Get the number of nodes from the adjacency matrix
    mfpt = mean_first_passage_time(adj) #Compute the mean first passage time matrix.  Assumes mean_first_passage_time is defined elsewhere.
    Ediff = 1.0 / mfpt #Compute the pairwise diffusion efficiency.  Set entries where i=j to zero.
    Ediff[np.diag_indices(n)] = 0  #Set diagonal elements to zero.
    GEdiff = np.sum(Ediff[~np.eye(n,dtype=bool)]) / (n**2 - n) #Compute the global mean diffusion efficiency.


    return GEdiff, Ediff

# Placeholder for the assumed mean_first_passage_time function
def mean_first_passage_time(adj):
    #This is a placeholder.  Replace with your actual implementation.
    n = adj.shape[0]
    mfpt = np.zeros((n,n))
    #Add your mean first passage time calculation here
    return mfpt

