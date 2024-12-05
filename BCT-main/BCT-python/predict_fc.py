# Translated from predict_fc.m

import numpy as np
from scipy.stats import pearsonr
from scipy.linalg import inv

def predict_fc(SC, FC, ED=None, pred_var=None, model='linear'):
    """
    Prediction of functional connectivity from structural connectivity

    Parameters
    ----------
    SC : numpy.ndarray
        Weighted/unweighted undirected NxN Structural Connectivity matrix.

    FC : numpy.ndarray
        Functional connections. FC can be a NxN symmetric matrix or an
        ((N*(N-1))/2) x 1 vector containing the upper triangular
        elements of the square FC matrix (excluding diagonal elements).

    ED : numpy.ndarray, optional
        Euclidean distance matrix or upper triangular vector of the
        matrix (optional).  Defaults to None.

    pred_var : list or numpy.ndarray, optional
        Set of M predictors. These can be given as an KxM array where
        K = ((N*(N-1))/2) and M is the number of predictors.
        Alternatively, pred_var can be a list with the names of network
        measures to be used as predictors. Accepted network measure
        names are:
            'SPLbin'        - Shortest-path length (binary)
            'SPLwei_inv'    - Shortest-path length computed with an inv transform
            'SPLwei_log'    - Shortest-path length computed with a log transform
            'SPLdist'       - Shortest-path length computed with no transform
            'SIbin'         - Search Information of binary shortest-paths
            'SIwei_inv'     - Search Information of shortest-paths computed with an inv transform
            'SIwei_log'     - Search Information of shortest-paths computed with a log transform
            'SIdist'        - Search Information of shortest-paths computed with no transform
            'T'             - Path Transitivity
            'deltaMFPT'     - Column-wise z-scored mean first passage time
            'neighOverlap'  - Neighborhood Overlap
            'MI'            - Matching Index

        If no predictors are specified, the defaults are ['SPLwei_log', 'SIwei_log']. Defaults to None.

    model : str, optional
        Specifies the order of the regression model. 'model' can be any option
        accepted by numpy.linalg.lstsq (e.g. 'linear', 'quadratic', etc.) If no model is specified,
        'linear' is the default.


    Returns
    -------
    FCpre : numpy.ndarray
        Predicted NxN Functional Connectivity matrix

    FCcorr : float
        Pearson Correlation between PCpred and FC

    beta : numpy.ndarray
        Regression Coefficients

    pred_data : numpy.ndarray
        KxM array of predictors.

    R : dict
        Dictionary containing regression results (beta, yhat, etc.).  Structure will differ from MATLAB's regstats output.

    References
    ----------
    Goñi et al (2014) PNAS,  833–838, doi: 10.1073/pnas.1315529111

    """

    pred_names = ['SPLbin','SPLwei_inv','SPLwei_log','SPLdist','SIbin',
                  'SIwei_inv','SIwei_log','SIdist','T','deltaMFPT','neighOverlap','MI']

    N = SC.shape[0]
    indx = np.triu_indices(N, k=1)

    # select model

    if pred_var is None and ED is not None:
        pred_var = ['ED','SPLwei_log','SI','T']
        flag_var_names = True
        flag_ED = True
    elif pred_var is None and ED is None:
        pred_var = ['SPLwei_log','SI','T']
        flag_var_names = True
        flag_ED = False
    elif pred_var is not None and isinstance(pred_var,list) and ED is not None:
        flag_var_names = True
        flag_ED = True
    elif pred_var is not None and isinstance(pred_var,list) and ED is None:
        flag_var_names = True
        flag_ED = False
    elif pred_var is not None and isinstance(pred_var, np.ndarray) and ED is not None:
        flag_var_names = False
        flag_ED = True
    elif pred_var is not None and isinstance(pred_var, np.ndarray) and ED is None:
        flag_var_names = False
        flag_ED = False
    else:
        raise ValueError('"pred_var" must be a KxM array of M predictors, or a list of graph-measure names.')


    if flag_ED:
        if ED.shape == (N,N):
            # square ED matrix
            pred_data = ED[indx]
        elif ED.shape == (len(indx[0]),1) or ED.shape == (len(indx[0]),):
            # ED is already an upper-triangle vector
            pred_data = ED
        else:
            raise ValueError('ED must be a square matrix or a vector containing the upper triangle of the square ED matrix')
    else:
        pred_data = np.empty((0,0))


    if flag_var_names:
        print('\n----------------------')
        print('\n Selected predictors: \n')
        ind2start = pred_data.shape[1]
        pred_data = np.concatenate((pred_data, np.zeros((len(indx[0]), len(pred_var)))), axis=1)

        for v, var in enumerate(pred_var):
            var_ind = pred_names.index(var)
            # placeholder functions - replace with actual implementations
            if var_ind == 0:  #SPLbin
                print('Shortest-path length (binary) \n\n')
                data = distance_wei_floyd(SC > 0, transform=None)
            elif var_ind == 1:   #SPLwei_inv
                print('Shortest-path length computed with an inv transform \n')
                data = distance_wei_floyd(SC, transform='inv')
            elif var_ind == 2:   #SPLwei_log
                print('Shortest-path length computed with a log transform \n')
                data = distance_wei_floyd(SC, transform='log')
            elif var_ind == 3:   #SPLdist
                print('Shortest-path length computed with no transform \n')
                data = distance_wei_floyd(SC, transform=None)
            elif var_ind == 4:   #SIbin
                print('Search Information of binary shortest-paths \n')
                data = search_information(SC > 0, transform=None)
                data = data + data.T
            elif var_ind == 5:   #SIwei_inv
                print('Search Information of shortest-paths computed with an inv transform \n')
                data = search_information(SC, transform='inv')
                data = data + data.T
            elif var_ind == 6:   #SIwei_log
                print('Search Information of shortest-paths computed with a log transform \n')
                data = search_information(SC, transform='log')
                data = data + data.T
            elif var_ind == 7:   #SIdist
                print('Search Information of shortest-paths computed with no transform \n')
                data = search_information(SC, transform=None)
                data = data + data.T
            elif var_ind == 8:   #T
                print('Path Transitivity \n')
                data = path_transitivity(SC > 0)
            elif var_ind == 9:  #deltaMFPT
                print('Column-wise z-scored mean first passage time \n')
                mfpt = mean_first_passage_time(SC)
                deltamfpt = (mfpt - np.mean(mfpt, axis=0)) / np.std(mfpt, axis=0)
                data = deltamfpt + deltamfpt.T
            elif var_ind == 10:  #neighOverlap
                print('Neighborhood Overlap \n')
                data = (SC > 0) @ (SC > 0).T
            elif var_ind == 11:  #MI
                print('Matching Index \n')
                data = matching_ind(SC)
            else:
                raise ValueError('This is not an accepted predictor. See list of available predictors')
            pred_data[:,ind2start+v] = data[indx]
    else:
        if pred_var.shape[0] == len(indx[0]):
            pred_data = np.concatenate((pred_data, pred_var), axis=1)
        else:
            raise ValueError('Custom predictors must be provided as KxM array of M predictors')


    if FC.shape == (N,N):
        # square FC matrix
        responses = FC[indx]
    elif FC.shape == (len(indx[0]),1) or FC.shape == (len(indx[0]),):
        # FC is already an upper-triangle vector
        responses = FC
    else:
        raise ValueError('FC must be a square matrix or a vector containing the upper triangle (no diagonal elements) of the square FC matrix')

    # run multilinear model using numpy.linalg.lstsq
    beta, _, _, _ = np.linalg.lstsq(pred_data, responses, rcond=None)
    
    #Regression results dictionary (simpler than MATLAB's regstats)
    R = {'beta': beta, 'yhat': pred_data @ beta}
    
    FCpre = np.zeros_like(SC)
    FCpre[indx] = R['yhat']
    FCpre = FCpre + FCpre.T
    FCcorr, _ = pearsonr(responses, FCpre[indx])

    return FCpre, FCcorr, beta, pred_data, R


# Placeholder functions - replace with your actual implementations

def distance_wei_floyd(W, transform='log'):
    """Calculates shortest path lengths. Replace with your actual implementation."""
    if transform == 'log':
        D = np.log(floyd_warshall(W))
    elif transform == 'inv':
        D = inv(floyd_warshall(W))
    else:
        D = floyd_warshall(W)
    return D

def floyd_warshall(W):
    """Calculates shortest paths using the Floyd-Warshall algorithm. Replace with your actual implementation."""
    n = W.shape[0]
    D = np.copy(W)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                D[i,j] = min(D[i,j],D[i,k]+D[k,j])
    return D


def search_information(W, transform='log'):
    """Calculates search information. Replace with your actual implementation."""
    if transform == 'log':
        SI = np.log(search_info(W))
    elif transform == 'inv':
        SI = inv(search_info(W))
    else:
        SI = search_info(W)
    return SI

def search_info(W):
    """Calculates search information. Replace with your actual implementation."""
    # Replace with your actual implementation
    return np.ones_like(W)


def path_transitivity(W):
    """Calculates path transitivity. Replace with your actual implementation."""
    # Replace with your actual implementation
    return np.ones_like(W)


def mean_first_passage_time(W):
    """Calculates mean first passage time. Replace with your actual implementation."""
    # Replace with your actual implementation
    return np.ones_like(W)


def matching_ind(W):
    """Calculates matching index. Replace with your actual implementation."""
    # Replace with your actual implementation
    return np.ones_like(W)


