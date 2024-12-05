# Translated from matching_ind.m

import numpy as np

def matching_ind(CIJ):
    #MATCHING_IND       Matching index
    #
    #   [Min,Mout,Mall] = matching_ind(CIJ);
    #
    #   For any two nodes u and v, the matching index computes the amount of
    #   overlap in the connection patterns of u and v. Self-connections and
    #   u-v connections are ignored. The matching index is a symmetric 
    #   quantity, similar to a correlation or a dot product.
    #
    #   Input:      CIJ,    connection/adjacency matrix
    #
    #   Output:     Min,    matching index for incoming connections
    #               Mout,   matching index for outgoing connections
    #               Mall,   matching index for all connections
    #
    #   Notes:
    #       Does not use self- or cross connections for comparison.
    #       Does not use connections that are not present in BOTH u and v.
    #       All output matrices are calculated for upper triangular only.
    #
    #
    # Olaf Sporns, Indiana University, 2002/2007/2008

    N = CIJ.shape[0]

    # compare incoming connections only
    Min = np.zeros((N,N))
    for i in range(N-1):
        for j in range(i+1,N):
            c1 = CIJ[:,i]
            c2 = CIJ[:,j]
            use = ~((~c1)&(~c2))
            use[i] = 0
            use[j] = 0
            ncon = np.sum(c1[use]) + np.sum(c2[use])
            if (ncon==0):
                Min[i,j] = 0
            else:
                Min[i,j] = 2*(np.sum(c1[use]&c2[use])/ncon)

    # compare outgoing connections only
    Mout = np.zeros((N,N))
    for i in range(N-1):
        for j in range(i+1,N):
            c1 = CIJ[i,:]
            c2 = CIJ[j,:]
            use = ~((~c1)&(~c2))
            use[i] = 0
            use[j] = 0
            ncon = np.sum(c1[use]) + np.sum(c2[use])
            if (ncon==0):
                Mout[i,j] = 0
            else:
                Mout[i,j] = 2*(np.sum(c1[use]&c2[use])/ncon)

    # compare all (incoming+outgoing) connections
    Mall = np.zeros((N,N))
    for i in range(N-1):
        for j in range(i+1,N):
            c1 = np.concatenate((CIJ[:,i],CIJ[i,:]))
            c2 = np.concatenate((CIJ[:,j],CIJ[j,:]))
            use = ~((~c1)&(~c2))
            use[i] = 0; use[i+N] = 0
            use[j] = 0; use[j+N] = 0
            ncon = np.sum(c1[use]) + np.sum(c2[use])
            if (ncon==0):
                Mall[i,j] = 0
            else:
                Mall[i,j] = 2*(np.sum(c1[use]&c2[use])/ncon)

    return Min, Mout, Mall

