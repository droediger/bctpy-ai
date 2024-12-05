# Translated from diversity_coef_sign.m

import numpy as np

def diversity_coef_sign(W, Ci):
    # Shannon-entropy based diversity coefficient
    #
    #   [Hpos Hneg] = diversity_coef_sign(W,Ci);
    #
    #   The Shannon-entropy based diversity coefficient measures the diversity
    #   of intermodular connections of individual nodes and ranges from 0 to 1.
    #
    #   Inputs:     W,      undirected connection matrix with positive and
    #                       negative weights
    #
    #               Ci,     community affiliation vector
    #
    #   Output:     Hpos,   diversity coefficient based on positive connections
    #               Hneg,   diversity coefficient based on negative connections
    #
    #   References: Shannon CE (1948) Bell Syst Tech J 27, 379-423.
    #               Rubinov and Sporns (2011) NeuroImage.
    #
    #
    #   2011-2012, Mika Rubinov, U Cambridge

    #   Modification History:
    #   Mar 2011: Original
    #   Sep 2012: Fixed treatment of nodes with no negative strength
    #             (thanks to Alex Fornito and Martin Monti)


    n = len(W)                                  #number of nodes
    m = np.max(Ci)                                    #number of modules

    Hpos = entropy(W*(W>0))
    Hneg = entropy(-W*(W<0))

    return Hpos, Hneg

def entropy(W_):
    S = np.sum(W_, axis=1)                          #strength
    Snm = np.zeros((len(W_),np.max(Ci)+1))                       #node-to-module degree
    for i in range(np.max(Ci)+1):                             #loop over modules
        Snm[:,i] = np.sum(W_[:,Ci==i+1],axis=1)
    pnm = Snm / S[:,None]
    pnm = np.nan_to_num(pnm)
    pnm[pnm==0] = 1
    H = -np.sum(pnm*np.log(pnm),axis=1)/np.log(np.max(Ci)+1)
    return H

