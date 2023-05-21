import stats as stats
import data_vis as visual
import data_ops as data
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

features, labels = data.read_labelled_data("./data/small_data.txt", "./data/small_labels.txt")

#feature selection 
#removing features with low variance : remove all features that are either 1/0 in more than 80% of samples

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

#dimension reduction 
#PCA -> aims to find the directions of maximum variance in high-dimensional data and projects it onto a new subspace with equal or fewer dimensions

def PCA_scratch(features,labels):
    #standardise the features
    sc = StandardScaler()
    X = sc.fit_transform(features)
    #covariance matrix
    cov_mat = np.cov(X.T)
    #perform eigendecomposition which should yield a vector (eigen_vals) consisting of 71 eigenvalues 
    # and corresponding eigenvectors as columns in 71x71 deimensional matrix (eigen_vecs)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    
    #calculate cumulative sum of explained variances
    tot = sum(eigen_vals)
    var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    
    #plot explained variances
    plt.bar(range(1,72), var_exp, alpha=0.5,
        align='center', label='individual explained variance')
    plt.step(range(1,72), cum_var_exp, where='mid',
            label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.show()
    
    #feature transformation
    
    #Make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    
    #sort the tuples from high to low
    eigen_pairs.sort(key = lambda k: k[0], reverse=True)
    
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
    print('Matrix W:\n', w)
    
    X[0].dot(w)
    X_pca = X.dot(w)
    
    colors = ['r', 'b', 'g','purple','orange']
    markers = ['s', 'x', 'o','^','p','*']
    for l, c, m in zip(np.unique(labels), colors, markers):
        plt.scatter(X[labels==l, 0], 
                    X_pca[labels==l, 1], 
                    c=c, label=l, marker=m) 
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.show()
    
    
PCA_scratch(features,labels)  

#Using the library
from sklearn.linear_model import LogisticRegression


#initialise pca and logistic regression model
pca = PCA(n_components=2)
lr = LogisticRegression(multi_class = 'auto', solver ='liblinear')
    
#fit and transform data
X_pca = pca.fit_transform(features)
lr.fit(X_pca, labels)
    
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=[cmap(idx)],
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)# plot decision regions for training set


plot_decision_regions(X_pca, labels, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

#clusters -> label is the cluster
    