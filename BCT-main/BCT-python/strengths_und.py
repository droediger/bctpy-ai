# Translated from strengths_und.m

import numpy as np

def strengths_und(CIJ):
    #STRENGTHS_UND        Strength
    #
    #   str = strengths_und(CIJ);
    #
    #   Node strength is the sum of weights of links connected to the node.
    #
    #   Input:      CIJ,    undirected weighted connection matrix
    #
    #   Output:     str,    node strength
    #
    #
    #   Olaf Sporns, Indiana University, 2002/2006/2008

    # compute strengths
    str = np.sum(CIJ, axis=1) # strength

    return str


