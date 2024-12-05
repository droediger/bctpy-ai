# Translated from grid_communities.m

def grid_communities(c):
    # GRID_COMMUNITIES       Outline communities along diagonal
    #
    #   [X, Y, INDSORT] = GRID_COMMUNITIES(C) takes a vector of community
    #   assignments C and returns three output arguments for visualizing the
    #   communities. The third is INDSORT, which is an ordering of the vertices
    #   so that nodes with the same community assignment are next to one
    #   another. The first two arguments are vectors that, when overlaid on the
    #   adjacency matrix using the matplotlib.pyplot.plot function, highlight the communities.
    #
    #   Example:
    #
    #   >> load AIJ;                                # load adjacency matrix
    #   >> [C,Q] = modularity_louvain_und(AIJ);     # get community assignments
    #   >> [X,Y,INDSORT] = fcn_grid_communities(C); # call function
    #   >> imagesc(AIJ(INDSORT,INDSORT));           # plot ordered adjacency matrix
    #   >> hold on;                                 # hold on to overlay community visualization
    #   >> plot(X,Y,'r','linewidth',2);             # plot community boundaries
    #
    #   Inputs:     C,       community assignments
    #
    #   Outputs:    X,       x coor
    #               Y,       y coor
    #               INDSORT, indices
    #
    #   Richard Betzel, Indiana University, 2012
    #

    import numpy as np

    nc = np.max(c)
    c_sorted, indsort = np.sort(c), np.argsort(c)

    X = []
    Y = []
    for i in range(1, nc + 1):
        ind = np.where(c_sorted == i)[0]
        if ind.size > 0:
            mn = np.min(ind) - 0.5
            mx = np.max(ind) + 0.5
            x = np.array([mn, mn, mx, mx, mn, np.nan])
            y = np.array([mn, mx, mx, mn, mn, np.nan])
            X = np.concatenate((X, x))
            Y = np.concatenate((Y, y))

    return X, Y, indsort


