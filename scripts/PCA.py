import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import data_ops as data
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#features, labels = data.read_labelled_data("./data/small_data.txt", "./data/small_labels.txt")
features, labels = data.read_labelled_data("./data/training_data.txt", "./data/training_labels.txt")

def using_PCA_library(features):
    #scale the data
    scalar = StandardScaler()
    features_scaled = scalar.fit_transform(features)
    print("Features Scaled: ")
    print(features_scaled)
    print(features_scaled.shape)
    #apply PCA

    pca = PCA()
    features_pca = pd.DataFrame(pca.fit_transform(features_scaled))
    print("Features PCA")
    print(features_pca)
    print(features_pca.shape) # with small data we have 10 principle components

    #varaince of each principle component
    exp_var_pca = pca.explained_variance_ratio_
    print("Explained variance")
    print(exp_var_pca)
    variance =0
    count=0
    
    for i in exp_var_pca:
        variance +=i
        count+=1
        if (variance >=0.8):
            break
    
    print(variance) 
    print("Amount of features that make up 0.8 of variance: ", count)
     
    #cumulate sum of eigenvalues : step plot for visualising the variance explained by each principal component
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)

    #visualisation plot
    plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    #plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    #check which features make up 80% of variances and add to new matrix (will be used for NN input)

def PCA_custom(features):
    #
    # Scale the dataset; This is very important before you apply PCA
    #
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(features)
    X_train_std = sc.transform(features)
   
    #
    # Import eigh method for calculating eigenvalues and eigenvectirs
    #
    from numpy.linalg import eigh
    #
    # Determine covariance matrix
    #
    cov_matrix = np.cov(X_train_std, rowvar=False)
    print("Covariance Matrix: ")
    print(pd.DataFrame(cov_matrix))
    print(cov_matrix.shape)
    #
    # Determine eigenvalues and eigenvectors
    #
    egnvalues, egnvectors = eigh(cov_matrix)
    #
    # Determine explained variance
    #
    total_egnvalues = sum(egnvalues)
    var_exp = [(i/total_egnvalues) for i in sorted(egnvalues, reverse=True)]
    var_exp_display = pd.DataFrame(var_exp)
    print("Variance Explained: ")
    print(var_exp_display)
    
    #
    # Plot the explained variance against cumulative explained variance
    #
    import matplotlib.pyplot as plt
    cum_sum_exp = np.cumsum(var_exp)
    plt.bar(range(0,len(var_exp)), var_exp, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0,len(cum_sum_exp)), cum_sum_exp, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    #plt.tight_layout()
    plt.show()

#PCA_custom(features)
using_PCA_library(features)





