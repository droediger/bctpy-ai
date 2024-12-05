# Translated from demo_generative_models_geometric.m

import numpy as np
import matplotlib.pyplot as plt

# Load data (assuming 'demo_generative_models_data.mat' is a .mat file that can be loaded using scipy.io.loadmat)
#  Replace this with your actual data loading method if necessary.  This assumes a structure with fields A, Aseed, D.
try:
    import scipy.io
    data = scipy.io.loadmat('demo_generative_models_data')
    A = data['A']
    Aseed = data['Aseed']
    D = data['D']
except FileNotFoundError:
    print("Error: 'demo_generative_models_data.mat' not found. Please provide the correct file.")
    exit()
except KeyError as e:
    print(f"Error: Field {e} not found in 'demo_generative_models_data.mat'. Please check your data file.")
    exit()



# Get cardinality of network
n = len(A)

# Set model type
modeltype = 'sptl'

# Set whether the model is based on powerlaw or exponentials
modelvar = [{'powerlaw'}, {'powerlaw'}]

# Choose some model parameters
nparams = 100
params = np.random.uniform(-10, 0, nparams)

# Generate synthetic networks and energy for the neighbors model;
#  Assume evaluate_generative_model is defined elsewhere and takes the specified inputs.
# Replace this with your actual implementation of evaluate_generative_model.

def evaluate_generative_model(Aseed,A,D,modeltype,modelvar,params):
    # Replace this with your actual implementation
    # This is a placeholder.  The original MATLAB code did not provide this function definition.
    return np.random.rand(nparams,3), np.random.rand(nparams,3), np.random.rand(nparams,3)

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

fig = plt.figure(figsize=(4, 4))  #Setting figure size in inches


for i in range(X.shape[1]):
    ax = fig.add_subplot(3, 2, i + 1)
    scatter = ax.scatter(params, X[:, i], s=100, c=X[:, i], cmap='jet')
    ax.set_ylim([0, 1])
    ax.set_clim([0, 1])
    ax.set_xlabel(r'geometric parameter, $\eta$')
    ax.set_ylabel(names[i][0])


plt.show()

