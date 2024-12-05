# Translated from cycprob.m

import numpy as np

def cycprob(Pq):
    # CYCPROB       Cycle probability
    #
    #   [fcyc,pcyc] = cycprob(Pq);
    #
    #   Cycles are paths which begin and end at the same node. Cycle 
    #   probability for path length d, is the fraction of all paths of length 
    #   d-1 that may be extended to form cycles of length d.
    #
    #   Input:      Pq,     3D numpy array, with Pq(i,j,q) = number of paths from 
    #                       'i' to 'j' of length 'q' (produced by 'findpaths')
    #
    #   Outputs:    fcyc,   fraction of all paths that are cycles for each path
    #                       length 'q'. 
    #               pcyc,   probability that a non-cyclic path of length 'q-1' 
    #                       can be extended to form a cycle of length 'q', for 
    #                       each path length 'q', 
    #
    #
    # Olaf Sporns, Indiana University, 2002/2007/2008

    # Note: fcyc(1) must be zero, as there cannot be cycles of length one.
    fcyc = np.zeros(Pq.shape[2])
    for q in range(Pq.shape[2]):
       if(np.sum(Pq[:,:,q])>0):
          fcyc[q] = np.trace(Pq[:,:,q])/np.sum(Pq[:,:,q])
       else:
          fcyc[q] = 0
    

    # Note: pcyc(1) is not defined (set to zero).
    # Note: pcyc(2) is equal to the fraction of reciprocal connections, 
    #       'frecip', delivered by 'reciprocal.m'.
    # Note: there are no non-cyclic paths of length N and no cycles of length N+1
    pcyc = np.zeros(Pq.shape[2])
    for q in range(1,Pq.shape[2]):
       if((np.sum(Pq[:,:,q-1])-np.trace(Pq[:,:,q-1]))>0):
          pcyc[q] = np.trace(Pq[:,:,q])/(np.sum(Pq[:,:,q-1])-np.trace(Pq[:,:,q-1]))
       else:
          pcyc[q] = 0

    return fcyc, pcyc

