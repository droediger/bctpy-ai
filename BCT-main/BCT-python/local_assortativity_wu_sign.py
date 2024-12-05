# Translated from local_assortativity_wu_sign.m

import numpy as np

def local_assortativity_wu_sign(W):
    #LOCAL_ASSORTATIVITY_WU_SIGN     Local Assortativity
    #
    #   [loc_assort_pos,loc_assort_neg] = local_assortativity_wu_sign(W);
    #
    #   Local Assortativity measures the extent to which nodes are connected to
    #   nodes of similar strength (vs. higher or lower strength). Adapted from
    #   Thedchanamoorthy et al. (2014)'s formula to allow weighted/signed 
    #   networks (node degree replaced with node strength). Note, output values 
    #   sum to total assortativity. 
    #
    #   Inputs:     W,        undirected connection matrix with positive and
    #                         negative weights
    #
    #   Output:     loc_assort_pos, local assortativity from positive weights
    #
    #               loc_assort_neg, local assortativity from negative weights
    #
    #   Reference: Thedchanamoorthy G, Piraveenan M, Kasthuriratna D, 
    #              Senanayake U. Proc Comp Sci (2014) 29:2449-2461.
    #
    #
    #   Jeff Spielberg, Boston University

    #   Modification History:
    #   May 2015: Original

    np.fill_diagonal(W, 0) #Set diagonal to zero
    r_pos = assortativity_wei(W * (W > 0), 0) #Assortativity from positive weights
    r_neg = assortativity_wei(-W * (W < 0), 0) #Assortativity from negative weights
    str_pos, str_neg = strengths_und_sign(W) #Calculate node strengths for positive and negative weights
    num_nodes = W.shape[0]
    loc_assort_pos = np.empty(num_nodes)
    loc_assort_neg = np.empty(num_nodes)
    loc_assort_pos[:] = np.nan
    loc_assort_neg[:] = np.nan

    for curr_node in range(num_nodes):
        j_pos = np.where(W[curr_node,:] > 0)[0] #Indices of nodes with positive connection to current node.
        if len(j_pos)>0:
            loc_assort_pos[curr_node] = np.sum(np.abs(str_pos[j_pos] - str_pos[curr_node])) / str_pos[curr_node]

        j_neg = np.where(W[curr_node,:] < 0)[0] #Indices of nodes with negative connection to current node.
        if len(j_neg)>0:
            loc_assort_neg[curr_node] = np.sum(np.abs(str_neg[j_neg] - str_neg[curr_node])) / str_neg[curr_node]

    loc_assort_pos = ((r_pos + 1) / num_nodes) - (loc_assort_pos / np.sum(loc_assort_pos))
    loc_assort_neg = ((r_neg + 1) / num_nodes) - (loc_assort_neg / np.sum(loc_assort_neg))

    return loc_assort_pos, loc_assort_neg

# Placeholder functions.  Replace with your actual implementations.
def assortativity_wei(W, directed):
    # Replace with your actual implementation of assortativity_wei
    # This is a placeholder
    return 0.5

def strengths_und_sign(W):
    # Replace with your actual implementation of strengths_und_sign
    # This is a placeholder
    return np.sum(np.abs(W), axis=1), np.sum(np.abs(W)*-1*(W<0), axis=1)

