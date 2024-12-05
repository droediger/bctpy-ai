# Translated from generate_fc.m

import numpy as np

def generate_fc(SC, beta, ED=None, pred_var=None, model='linear', FC=None):
    """GENERATE_FC     Generation of synthetic functional connectivity matrices

    [FCpre,pred_data,Fcorr] = generate_fc(SC,beta,ED,{'SPLwei_log','SIwei_log'},FC)
    [FCpre,pred_data] = generate_fc(SC,beta,[],{'SPLwei_log','SIwei_log'})

    Uses a vector beta of regression coefficients from the model
    FC = pred_var*beta to predict FC. pred_var are structural-based network
    measures derived from the structural connectivity network.

    Inputs:

        SC,
            Weighted/unweighted undirected NxN Structural Connectivity matrix.

        beta,
            Regression coefficients (vector). These may be obtained as an
            output parameter from function predict_fc.m

        ED,
            Euclidean distance matrix or upper triangular vector of the
            matrix (optional)

        pred_var,
            Set of M predictors. These can be given as an KxM array where
            K = ((N*(N-1))/2) and M is the number of predictors.
            Alternatively, pred_var can be a list with the names of network
            measures to be used as predictors. Accepted network measure
            names are:
                SPLbin        - Shortest-path length (binary)
                SPLwei_inv    - Shortest-path length computed with an inv transform
                SPLwei_log    - Shortest-path length computed with a log transform
                SPLdist       - Shortest-path length computed with no transform
                SIbin         - Search Information of binary shortest-paths
                SIwei_inv     - Search Information of shortest-paths computed with an inv transform
                SIwei_log     - Search Information of shortest-paths computed with a log transform
                SIdist        - Search Information of shortest-paths computed with no transform
                T             - Path Transitivity
                deltaMFPT     - Column-wise z-scored mean first passage time
                neighOverlap  - Neighborhood Overlap
                MI            - Matching Index

            Predictors must be specified in the order that matches the
            given beta values.

        model,
            Specifies the order of the regression model used. 'model' can be any option
            'linear'. If no model is specified, 'linear' is the default.

        FC,
            Functional connections. FC can be a NxN symmetric matrix or a
            ((N*(N-1))/2) x 1 vector containing the upper triangular
            elements of the square FC matrix (excluding diagonal elements).
            This argument is optional and only used to compute the
            correlation between the predicted FC and empirical FC.


    Outputs:

        FCpre,
            Predicted NxN Functional Connectivity matrix

        pred_data,
            KxM array of predictors.

        Fcorr,
            Pearson Correlation between FCpred and FC


    Reference: Goñi et al. (2014) PNAS 111: 833–838

    Andrea Avena-Koenigsberger, Joaquin Goñi and Olaf Sporns; IU Bloomington, 2016
    """

    beta = np.atleast_2d(beta).T # beta must be a column vector

    pred_names = ['SPLbin','SPLwei_inv','SPLwei_log','SPLdist','SIbin',
                  'SIwei_inv','SIwei_log','SIdist','T','deltaMFPT','neighOverlap','MI']

    N = SC.shape[0]
    indx = np.triu_indices(N, k=1)

    flag_var_names = False
    flag_ED = False

    if pred_var is None:
        if ED is not None:
            pred_var = ['ED','SPLwei_log','SI','T']
            flag_var_names = True
            flag_ED = True
        else:
            pred_var = ['SPLwei_log','SI','T']
            flag_var_names = True
    elif isinstance(pred_var, (list,tuple)):
        flag_var_names = True
        flag_ED = (ED is not None)
    elif isinstance(pred_var, np.ndarray):
        flag_var_names = False
        flag_ED = (ED is not None)
    else:
        raise ValueError('"pred_var" must be a KxM array of M predictors, or a list of graph-measure names.')


    if flag_ED:
        if ED.shape == (N,N):
            pred_data = ED[indx]
        elif ED.shape == (len(indx[0]),1):
            pred_data = ED.flatten()
        else:
            raise ValueError('ED must be a square matrix or a vector containing the upper triangle of the square ED matrix')
    else:
        pred_data = np.empty((0,0))

    if flag_var_names:
        print('\n----------------------')
        print('\n Selected predictors: ')
        pred_data = np.concatenate((pred_data, np.zeros((len(indx[0]),len(pred_var)))), axis=1)

        for i, v in enumerate(pred_var):
            var_ind = pred_names.index(v)
            if var_ind == 0:  #SPLbin
                print('Shortest-path length (binary) \n\n')
                data = distance_wei_floyd(SC > 0, 'binary')
            elif var_ind == 1:  #SPLwei_inv
                print('Shortest-path length computed with an inv transform \n')
                data = distance_wei_floyd(SC, 'inv')
            elif var_ind == 2:  #SPLwei_log
                print('Shortest-path length computed with a log transform \n')
                data = distance_wei_floyd(SC, 'log')
            elif var_ind == 3:  #SPLdist
                print('Shortest-path length computed with no transform \n')
                data = distance_wei_floyd(SC)
            elif var_ind == 4:  #SIbin
                print('Search Information of binary shortest-paths \n')
                data = search_information(SC > 0, 'binary')
                data = data + data.T
            elif var_ind == 5:  #SIwei_inv
                print('Search Information of shortest-paths computed with an inv transform \n')
                data = search_information(SC, 'inv')
                data = data + data.T
            elif var_ind == 6:  #SIwei_log
                print('Search Information of shortest-paths computed with a log transform \n')
                data = search_information(SC, 'log')
                data = data + data.T
            elif var_ind == 7:  #SIdist
                print('Search Information of shortest-paths computed with no transform \n')
                data = search_information(SC)
                data = data + data.T
            elif var_ind == 8:  #T
                print('Path Transitivity \n')
                data = path_transitivity(SC > 0, 'binary')
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
            pred_data[:, i + pred_data.shape[1] - len(pred_var)] = data[indx]

    elif pred_var.shape[0] == len(indx[0]):
        pred_data = np.concatenate((pred_data, pred_var), axis=1)
    else:
        raise ValueError('Custom predictors must be provided as a KxM array of M predictors')


    pred_data = x2fx(pred_data, model)

    if pred_data.shape[1] == beta.shape[0]:
        Y = pred_data @ beta
        FCpre = np.zeros((N, N))
        FCpre[indx] = Y
        FCpre = FCpre + FCpre.T
    else:
      FCpre = np.array([])


    if FC is not None:
        if FC.shape == (N, N):
            FCemp = FC[indx]
        elif FC.shape == (len(indx[0]), 1):
            FCemp = FC.flatten()
        else:
            print('Warning: FC must be a square matrix or a vector containing the upper triangle (no diagonal elements) of the square FC matrix')
            Fcorr = np.nan
        else:
            Fcorr = np.corrcoef(Y, FCemp)[0, 1]
    else:
        Fcorr = np.array([])


    return FCpre, pred_data, Fcorr



# Placeholder functions - replace with your actual implementations
def distance_wei_floyd(W, method=''):
    # Placeholder for distance_wei_floyd function
    if method == 'binary':
        return np.array([[1,2,3],[2,1,4],[3,4,1]])
    elif method == 'inv':
        return np.array([[1,2,3],[2,1,4],[3,4,1]])
    elif method == 'log':
        return np.array([[1,2,3],[2,1,4],[3,4,1]])
    else:
        return np.array([[1,2,3],[2,1,4],[3,4,1]])


def search_information(W, method=''):
    # Placeholder for search_information function
    if method == 'binary':
        return np.array([[1,2,3],[2,1,4],[3,4,1]])
    elif method == 'inv':
        return np.array([[1,2,3],[2,1,4],[3,4,1]])
    elif method == 'log':
        return np.array([[1,2,3],[2,1,4],[3,4,1]])
    else:
        return np.array([[1,2,3],[2,1,4],[3,4,1]])

def path_transitivity(W, method=''):
    # Placeholder for path_transitivity function
    return np.array([[1,2,3],[2,1,4],[3,4,1]])

def mean_first_passage_time(W):
    # Placeholder for mean_first_passage_time function
    return np.array([[1,2,3],[2,1,4],[3,4,1]])

def matching_ind(W):
    # Placeholder for matching_ind function
    return np.array([[1,2,3],[2,1,4],[3,4,1]])


def x2fx(X, model):
    # Placeholder for x2fx function - assumes linear model
    return X

