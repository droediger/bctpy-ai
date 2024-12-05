# Translated from strengths_dir.m

import numpy as np

def strengths_dir(CIJ):
    # Node strength is the sum of weights of links connected to the node. The
    # instrength is the sum of inward link weights and the outstrength is the
    # sum of outward link weights.

    # Input:      CIJ,    directed weighted connection matrix

    # Output:     is,     node instrength
    #               os,     node outstrength
    #               str,    node strength (instrength + outstrength)

    # Notes:  Inputs are assumed to be on the columns of the CIJ matrix.

    # compute strengths
    is = np.sum(CIJ, axis=0)  # instrength = column sum of CIJ
    os = np.sum(CIJ, axis=1)  # outstrength = row sum of CIJ
    str = is + os           # strength = instrength + outstrength

    return is, os, str


