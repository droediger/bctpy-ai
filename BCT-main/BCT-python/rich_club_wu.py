# Translated from rich_club_wu.m

import numpy as np

def rich_club_wu(CIJ, *varargin):
    """
    Rich club coefficients curve (weighted undirected graph)

    Rw = rich_club_wu(CIJ,varargin) % rich club curve for weighted graph

    The weighted rich club coefficient, Rw, at level k is the fraction of
    edge weights that connect nodes of degree k or higher out of the
    maximum edge weights that such nodes might share.

    Inputs:
        CIJ:        weighted directed connection matrix

        k-level:    (optional) max level of RC(k).
                    (by default k-level equals the maximal degree of CIJ)

    Output:
        Rw:         rich-club curve

    References:
        T Opsahl et al. Phys Rev Lett, 2008, 101(16)
        M van den Heuvel, O Sporns, J Neurosci 2011 31(44)

    Martijn van den Heuvel, University Medical Center Utrecht, 2011

    Modification History:
    2011: Original
    2015: Expanded documentation (Mika Rubinov)
    """

    NofNodes = CIJ.shape[1]  #number of nodes
    NodeDegree = np.sum(CIJ != 0, axis=1) #define degree of each node

    #define to which level rc should be computed
    if len(varargin) == 1:
        klevel = varargin[0]
    elif len(varargin) == 0:
        klevel = np.max(NodeDegree)
    else:
        raise ValueError('number of inputs incorrect. Should be [CIJ], or [CIJ, klevel]')

    #wrank contains the ranked weights of the network, with strongest connections on top
    wrank = np.sort(CIJ.flatten())[::-1]

    Rw = np.zeros(klevel)
    #loop over all possible k-levels
    for kk in range(1, klevel + 1):

        SmallNodes = np.where(NodeDegree < kk)[0]

        if SmallNodes.size == 0:
            Rw[kk - 1] = np.nan
            continue

        #remove small nodes with NodeDegree<kk
        CutoutCIJ = np.delete(np.delete(CIJ, SmallNodes, axis=0), SmallNodes, axis=1)

        #total weight of connections in subset E>r
        Wr = np.sum(CutoutCIJ)

        #total number of connections in subset E>r
        Er = np.sum(CutoutCIJ != 0)

        #E>r number of connections with max weight in network
        wrank_r = wrank[:Er]

        #weighted rich-club coefficient
        Rw[kk - 1] = Wr / np.sum(wrank_r)

    return Rw

