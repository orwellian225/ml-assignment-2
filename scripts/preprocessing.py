import stats as stats
import data_vis as visual
import data_ops as data
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import matplotlib.pyplot as plt

features, labels = data.read_labelled_data("./data/small_data.txt", "./data/small_labels.txt")

#dimension reduction
#removing features with low variance : remove all features that are either 1/0 in more than 80% of samples

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



#PCA


#clusters -> label is the cluster
    