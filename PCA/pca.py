import numpy as np
from sklearn.decomposition import PCA

###############################################################
# Applying PCA to get the low rank approximation (For Part 1) #
###############################################################

def pca_approx(M, m=100):
    '''
    Inputs:
        - M: The co-occurrence matrix (3,000 x 3,000)
        - m: The number of principal components we want to find
    Return:
        - Mc: The centered log-transformed covariance matrix (3,000 x 3,000)
        - V: The matrix containing the first m eigenvectors of Mc (3,000 x m)
        - eigenvalues: The array of the top m eigenvalues of Mc sorted in decreasing order
        - frac_var: |Sum of top m eigenvalues of Mc| / |Sum of all eigenvalues of Mc|
    '''
    np.random.seed(12) # DO NOT CHANGE THE SEED
    #####################################################################################################################################
    # TODO: Implement the following steps:
    # i) Apply log transformation on M to get M_tilde, such that M_tilde[i,j] = log(1+M[i,j]).
    # ii) Get centered M_tilde, denoted as Mc. First obtain the (d-dimensional) mean feature vector by averaging across all datapoints (rows).
    # Then subtract it from all the n feature vectors. Here, n = d = 3,000.
    # iii) Use the PCA function (fit method) from the sklearn library to apply PCA on Mc and get its rank-m approximation (Go through
    # the documentation available at: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
    # iv) Return the centered matrix, set of principal components (eigenvectors), eigenvalues, and fraction of variance explained by the
    # first m eigenvectors. Note that the values returned by the function should be in the order mentioned above and make sure all the 
    # dimensions are correct (apply transpose, if required).
    #####################################################################################################################################
    M_tilde = np.log(1+M)
    
    mean_feat_vec = np.mean(M_tilde,axis=0)
    Mc = M_tilde - mean_feat_vec
    
    pca = PCA(n_components=m)
    pca.fit(Mc)
    V = pca.components_.T
    eigenvalues = pca.explained_variance_
    
    frac_var = np.sum(pca.explained_variance_ratio_)
    
    return Mc, V, eigenvalues, frac_var

####################################################
# Get the Word Embeddings (For Parts 2, 3, 4, 5, 6)#
####################################################

def compute_embedding(Mc, V):
    '''
    Inputs:
        - Mc: The centered covariance matrix (3,000 x 3,000)
        - V: The matrix containing the first m eigenvectors of Mc (3,000 x m)
    Return:
        - E: The embedding matrix (3,000 x m), where m = length of embeddings
    '''
    #####################################################################################################################
    # TODO: Implement the following steps:
    # i) Get P = McV. Normalize the columns of P (to have unit l2-norm) to get E.
    # ii) Normalize the rows of E to have unit l2-norm and return it. This will be used in Parts 2, 4, 5, 6.
    #####################################################################################################################
    P = Mc @ V
    P /= np.linalg.norm(P,axis=0,keepdims=True)
    
    E = P / np.linalg.norm(P, axis=1, keepdims=True)
    
    return E