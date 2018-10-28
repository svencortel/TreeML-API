
# coding: utf-8

# In[1]:


from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.datasets import load_iris
from collections import defaultdict
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction.text import CountVectorizer
import time
from Tree import *


# In[2]:


# Load iris data-set
iris=load_iris()
X=iris.data
y=iris.target
# X_Final = []
# y_Final = []
# # Filter the data with 2 labels
# for idx_of_label, val_of_label in enumerate(y):
#     if(val_of_label != 2):
#         X_Final.append(X[idx_of_label])
#         y_Final.append(val_of_label)
X_Final = np.array(X)
y_Final = np.array(y)


print(X_Final, y_Final)

clf = DecisionTreeClassifier(max_depth=1)
clf.fit(X_Final, y_Final)
print(clf.predict(X_Final))

st = time.time()
# Main method:  All the processing happens here
print(mutual_info_classif(X_Final[:,1].reshape(-1,1), y_Final))
print(X_Final.dtype)
predictedLabelsByPerceptron = getDataSplit(X_Final[:,0].reshape(-1,1), y_Final)
labelIndicesOfPredictedData = SplitMasterDataByLabels(predictedLabelsByPerceptron, X_Final, y_Final)
print(labelIndicesOfPredictedData["leftExamples"])
print(labelIndicesOfPredictedData["leftLabels"])
print(labelIndicesOfPredictedData["rightLabels"])
predictedLabelsByPerceptronForLeftSplit = getDataSplit(labelIndicesOfPredictedData["leftExamples"][:,0].reshape(-1,1),
                                                       labelIndicesOfPredictedData["leftLabels"])
print(predictedLabelsByPerceptronForLeftSplit)
en = time.time()

print(en-st)
