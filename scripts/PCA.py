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

    pca = PCA(n_components=53)
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




def myplot(score, coeff, labels=None):
        xs = score[:,0]
        ys = score[:,1]
        n = coeff.shape[0]
        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())
        plt.scatter(xs * scalex,ys * scaley, c = labels)
        for i in range(n):
            plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
            if labels is None:
                plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
            else:
                plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'b', ha = 'center', va = 'center')
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.xlabel("PC{}".format(1))
        plt.ylabel("PC{}".format(2))
        plt.grid()
        
def PCA_analysis(features,labels):
    y = labels
    scalar = StandardScaler()
    features_scaled = scalar.fit_transform(features)
    
    pca = PCA()
    features_new = pca.fit_transform(features_scaled)
    
    myplot(features_new[:,0:2], np.transpose(pca.components_[0:2, :]))
    plt.show()
    
    print("Explained Variance")
    print(pca.explained_variance_ratio_)
    
    print()
    print("Components")
    print(abs(pca.components_))
    

def important_features(features):
    
    #z-score the features
    scalar = StandardScaler()
    scalar.fit(features)
    features = scalar.transform(features)
    
    #PCA model
    pca = PCA()
    feature_new = pca.fit_transform(features)
    
    #plotting data before and after PCA transform
    fig, axes = plt.subplots(1,2)
    axes[0].scatter(features[:,0], features[:,1], c=labels)
    axes[0].set_xlabel('x1')
    axes[0].set_ylabel('x2')
    axes[0].set_title('Before PCA')
    axes[1].scatter(feature_new[:,0], feature_new[:,1], c=labels)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title('After PCA')
    plt.show()
    
    print(pca.explained_variance_ratio_)
    
  
    #number of components
    n_pcs = pca.components_.shape[0]
    print(n_pcs)
    
    #get index of most important feature on Each component
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    
    initial_features = []
    for i in range(0,72):
        initial_features.append(i)
    
    most_important_names = [initial_features[most_important[i]] for i in range(n_pcs)]
    
    dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

    # build the dataframe
    df = pd.DataFrame(dic.items())
    print("most important features")
    print(df)
    
    
#important_features(features)


#PCA_analysis(features,labels)


