# Translated from makerandCIJdegreesfixed.m

import numpy as np

def makerandCIJdegreesfixed(in_degree, out_degree):
    """
    Generates a directed random network with specified in-degree and out-degree sequences.

    Args:
        in_degree (numpy.ndarray): In-degree vector.
        out_degree (numpy.ndarray): Out-degree vector.

    Returns:
        tuple: A tuple containing the binary directed connectivity matrix (cij) and a flag 
               (flag). flag=1 if the algorithm succeeded; flag=0 otherwise.
    """

    # Initialize
    n = len(in_degree)
    k = np.sum(in_degree)
    inInv = np.zeros(k, dtype=int)
    outInv = np.zeros(k, dtype=int)
    iIn = 0
    iOut = 0

    for i in range(n):
        inInv[iIn:iIn + in_degree[i]] = i
        outInv[iOut:iOut + out_degree[i]] = i
        iIn += in_degree[i]
        iOut += out_degree[i]

    cij = np.eye(n)
    edges = np.vstack((outInv, np.random.permutation(inInv)))

    # Create cij, and check for double edges and self-connections
    for i in range(k):
        if cij[edges[0, i], edges[1, i]]:
            warningCounter = 1
            while True:
                switchTo = np.random.randint(k)
                if not (cij[edges[0, i], edges[1, switchTo]] or cij[edges[0, switchTo], edges[1, i]]):
                    cij[edges[0, i], edges[1, switchTo]] = 1
                    if switchTo < i:
                        cij[edges[0, switchTo], edges[1, switchTo]] = 0
                        cij[edges[0, switchTo], edges[1, i]] = 1
                    temp = edges[1, i]
                    edges[1, i] = edges[1, switchTo]
                    edges[1, switchTo] = temp
                    break
                warningCounter += 1
                # If there is a legitimate substitution, it has a probability of 1/k of being done.
                # Thus it is highly unlikely that it will not be done after 2*k^2 attempts.
                # This is an indication that the given indegree / outdegree vectors may not be possible.
                if warningCounter == 2 * k**2:
                    flag = 0  # No valid solution found
                    return cij, flag

        else:
            cij[edges[0, i], edges[1, i]] = 1

    cij = cij - np.eye(n)

    # A valid solution was found
    flag = 1
    return cij, flag

