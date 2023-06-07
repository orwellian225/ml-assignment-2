import stats as stats
import data_ops as data
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import data_ops as data
from sklearn.preprocessing import StandardScaler


#features, labels = data.read_labelled_data("./data/small_data.txt", "./data/small_labels.txt")
features, labels = data.read_labelled_data("./data/training_data.txt", "./data/training_labels.txt")

    

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
    print(cov_matrix)
    #
    # Determine eigenvalues and eigenvectors
    # eigenvectors represent the principle components that contain most of the information (variance) represented using features
    #
    eigenDecom = eigh(cov_matrix)
    egnvalues = eigenDecom[0]
    egnvectors = eigenDecom[1]
    
    idx = np.argsort(egnvalues)[::-1]
    egnvalues = egnvalues[idx]
    egnvectors = egnvectors[:,idx]
    
    #
    # Determine explained variance
    #
    total_egnvalues = sum(egnvalues)
    #var_exp = [(i/total_egnvalues) for i in sorted(egnvalues, reverse=True)]
    var_exp = [(i/total_egnvalues) for i in egnvalues]
    var_exp_display = pd.DataFrame(var_exp)
    print("Variance Explained: ")
    print(var_exp_display)
    
    variance =0
    count=0
    for i in var_exp:
        variance +=i
        count+=1
        if (variance >=0.8):
            break
    
    print(variance) 
    print("Amount of features that make up 0.8 of variance: ", count)
    
    #Forming feature matrix
    
    dimensions = count+1
    adjustedData = X_train_std
    
    featureMatrix = egnvectors[:,1:dimensions]
    print(featureMatrix)
    #deriving new dataSet
    finalData = np.matmul(np.transpose(featureMatrix),np.transpose(adjustedData))
    finalData = np.transpose(finalData)
    print(pd.DataFrame(finalData))
    
<<<<<<< Updated upstream
=======
    file = open('./data/pca_matrix.txt', 'w')
    np.savetxt(file, featureMatrix)
    file.close()
    
    updateFile = open('./data/pca_matrix.txt', 'r+')
    input = updateFile.read()
    input = input.replace(' ', ',')
    newFile = open('./data/pca_matrix.txt','w+')
    newFile.write(input)
    updateFile.close()
    newFile.close()
>>>>>>> Stashed changes
    # Plot the explained variance against cumulative explained variance
    #
    import matplotlib.pyplot as plt
    cum_sum_exp = np.cumsum(var_exp)
    plt.bar(range(0,len(var_exp)), var_exp, alpha=0.5, align='center', label='Individual explained variance')
    #plt.step(range(0,len(cum_sum_exp)), cum_sum_exp, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    #plt.tight_layout()
    plt.show()
    
   


PCA_custom(features)


    
#selects best features based on univariate statistical tests
def univariate_feature_selection():
    selector = SelectKBest(f_classif, k=4) #select the 4 most significant features
    selector.fit(features, labels)
                
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()

    X_indices = np.arange(features.shape[-1])
    plt.figure(1)
    plt.clf()
    plt.bar(X_indices - 0.05, scores, width=0.2)
    plt.title("Feature univariate score")
    plt.xlabel("Feature number")
    plt.ylabel(r"Univariate score ($-Log(p_{value})$)")
    plt.show()

univariate_feature_selection()


    

    