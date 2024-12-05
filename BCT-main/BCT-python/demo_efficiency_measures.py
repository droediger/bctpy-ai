# Translated from demo_efficiency_measures.m

import numpy as np

# Assume distance_wei_floyd, rout_efficiency, diffusion_efficiency, and resource_efficiency_bin are defined elsewhere.
# These functions would need to be implemented based on their MATLAB counterparts.

# This script loads 7 unweighted, undirected adjacency matrices
# corresponding to a clique, chain, ring, 1D lattice, star, rich-club, and
# bi-modular toy networks (all graphs have 50 nodes). The following
# efficiency measures are computed for each graph:
#
#  - prob_SPL: probability of one particle traveling through shortest-paths
#  - Erout: efficiency of routing -> based on shortest-paths
#  - Ediff: efficiency of diffusion -> based on mean-first-passage-times
#  - Eres: efficiency of resources -> based on number of particles
#    necessary so that at least one particle taking shortest-paths with
#    certain probability (lambda).
#
#  If you are using this efficiency package for your research, please kindly
#  cite the paper:
#
#  "Exploring the Morphospace of Communication Efficiency in Complex
#  Networks" Goñi J, Avena-Koenigsberger A, Velez de Mendizabal N, van den
#  Heuvel M, Betzel RF and Sporns O. PLoS ONE. 2013
#
#  These examples and results correspond to Table 1 in the paper.
#
#  Joaquin Goñi and Andrea Avena-Koenigsberger, IU Bloomington, 2012


# Assume adjacency matrices are loaded from a file or are defined elsewhere.  Replace with your actual loading mechanism.
# For demonstration, let's use placeholders
clique = np.random.randint(0, 2, size=(50, 50))
chain = np.random.randint(0, 2, size=(50, 50))
ring = np.random.randint(0, 2, size=(50, 50))
lattice1D = np.random.randint(0, 2, size=(50, 50))
star = np.random.randint(0, 2, size=(50, 50))
rich_club = np.random.randint(0, 2, size=(50, 50))
bi_modular = np.random.randint(0, 2, size=(50, 50))

lambda_param = 0.5  # this parameter is an input for the computation of Eres.

# run and display efficiency measures for the 7 graphs
print('    prob_SPL  ', '  Erout  ', '  Ediff  ', '  Eres  ')

graphs = {
    'clique': clique,
    'chain': chain,
    'ring': ring,
    'lattice1D': lattice1D,
    'star': star,
    'rich_club': rich_club,
    'bi-modular': bi_modular
}

for graph_name, adj in graphs.items():
    print(f'----- {graph_name} ----- \n')
    N = adj.shape[0]
    EYE = np.eye(N, dtype=bool)
    SPL = distance_wei_floyd(adj) # Placeholder function call
    Erout = rout_efficiency(adj) # Placeholder function call
    Ediff = diffusion_efficiency(adj) # Placeholder function call
    Eres, prob_SPL = resource_efficiency_bin(adj, lambda_param, SPL) # Placeholder function call
    prob_SPL = np.mean(prob_SPL[~EYE])
    Eres = np.mean(Eres[~EYE])
    print([prob_SPL, Erout, Ediff, Eres])


