# Translated from writetoPAJ.m

import numpy as np

def writetoPAJ(CIJ, fname, arcs):
    # WRITETOPAJ         Write to Pajek
    #
    #   writetoPAJ(CIJ, fname, arcs);
    #
    #   This function writes a Pajek .net file from a NumPy array
    #
    #   Inputs:     CIJ,        adjacency matrix (NumPy array)
    #               fname,      filename minus .net extension (string)
    #               arcs,       1 for directed network, 0 for undirected network (integer)
    #
    #   Adapted from Chris Honey, Indiana University, 2007

    N = CIJ.shape[0]  #Get the number of rows in the matrix.
    fname_ext = fname + ".net" #Create the full filename.

    fid = open(fname_ext, 'w') #Open the file for writing.

    # VERTICES
    fid.write("*vertices %6i \r\n" % N) #Write the number of vertices.
    for i in range(1, N + 1):
        fid.write("%6i \"%6i\" \r\n" % (i, i)) #Write each vertex.


    # ARCS/EDGES
    if arcs:
        fid.write("*arcs \r\n") #Write the header for arcs.
    else:
        fid.write("*edges \r\n") #Write the header for edges.

    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if CIJ[i - 1, j - 1] != 0:  #Check if the element is non-zero. Adjust indices for 0-based indexing in Python.
                fid.write("%6i %6i %6f \r\n" % (i, j, CIJ[i - 1, j - 1])) #Write the arc or edge.

    fid.close() #Close the file.



