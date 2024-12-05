# Translated from demo_generative_models_neighbors.m

import numpy as np
import matplotlib.pyplot as plt

# Load data (assuming 'demo_generative_models_data.mat' is a NumPy-compatible file)
data = np.load('demo_generative_models_data.npz')
A = data['A']
Aseed = data['Aseed']
D = data['D']

# Get cardinality of the network
n = len(A)

# Set model type
modeltype = 'matching'

# Set whether the model is based on powerlaw or exponentials
modelvar = [{'powerlaw'}, {'powerlaw'}]

# Choose some model parameters
nparams = 100
params = np.column_stack((np.random.uniform(-10, 0, nparams), np.random.uniform(-1, 1, nparams)))

# Generate synthetic networks and energy for the neighbors model (assuming evaluate_generative_model is defined elsewhere)
B, E, K = evaluate_generative_model(Aseed, A, D, modeltype, modelvar, params)  
X = np.column_stack((E, K))

# Show scatterplot of parameter values versus energy and KS statistics
names = [
    ['energy'],
    ['degree'],
    ['clustering'],
    ['betweenness'],
    ['edge length']
]

# Create figure
plt.figure(figsize=(4, 4))

for i in range(X.shape[1]):
    plt.subplot(3, 2, i + 1)
    plt.scatter(params[:, 0], params[:, 1], s=100, c=X[:, i], cmap='jet')
    plt.clim(0, 1)
    plt.xlabel('geometric parameter, \\eta')
    plt.ylabel('topological parameter, \\gamma')
    plt.title(names[i][0])

plt.tight_layout()
plt.show()



