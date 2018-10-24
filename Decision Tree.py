
# coding: utf-8

# In[1]:


from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import numpy as np
from sklearn.datasets import load_digits
from collections import defaultdict
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


# Load iris data-set
iris=load_iris()
X=iris.data
y=iris.target
X_Final = []
y_Final = []
# Filter the data with 2 labels
for idx_of_label, val_of_label in enumerate(y):
    if(val_of_label != 2):
        X_Final.append(X[idx_of_label])
        y_Final.append(val_of_label) 
X_Final = np.array(X_Final)
y_Final = np.array(y_Final)


# In[8]:


# Process each feature using the perceptron model
# Input: feature data, labels
# Return: Predicted value by the model

def ProcessFeatureByPerceptronModel (samples, labels):
    clf = Perceptron(random_state=0)
    clf.fit(samples,labels)
    cv = CountVectorizer(max_df=0.95, min_df=2,
                                     max_features=10000,
                                     stop_words='english')
    print(clf.score(samples, labels))    
    return clf.predict(samples)


# In[9]:


# Get the feature data from a data-set for a provided feature index
# Input: feature data, feature index for which data needs to be extracted
# Return: Data in the given feature

def GetFeatureData(samples,featureIndex):
    feature_data = []
    for idx, data in enumerate(samples):  
        array = []
        array.append(samples[idx][featureIndex])
        feature_data.append(array)
    return feature_data


# In[10]:


# Split master data by predicted labels by the model
# Input: predicted labels
# Return: Dictionary with the left and right sub-sets (split the master data)

def SplitMasterDataByLabels(labels):
    uniqueLabels = set(labels)
    dictionaryLabels = defaultdict(list)
    for unique in uniqueLabels:
        indices = [i for i, x in enumerate(labels) if x == unique]
        dictionaryLabels[unique].append(indices)
    splittedData = defaultdict()
    splittedData["leftExamples"] = FilterDataByIndices(X_Final,list(dictionaryLabels.values())[0])
    splittedData["leftLabels"] = FilterDataByIndices(y_Final,list(dictionaryLabels.values())[0])
    splittedData["rightExamples"] = FilterDataByIndices(X_Final,list(dictionaryLabels.values())[1])
    splittedData["rightLabels"] = FilterDataByIndices(y_Final,list(dictionaryLabels.values())[1])      
    return splittedData


# In[11]:


# Get instances for the list of row indexes
# Input: sample data, list of row index
# Return: data instances at the provided index

def FilterDataByIndices(data, indexList):
    for index in indexList:
            tempDataList = data[index]           
    return tempDataList    


# In[12]:


# Main method:  All the processing happens here
predictedLabelsByPerceptron = ProcessFeatureByPerceptronModel(X_Final, y_Final)
labelIndicesOfPredictedData = SplitMasterDataByLabels(predictedLabelsByPerceptron)
print(labelIndicesOfPredictedData["leftExamples"])
print(labelIndicesOfPredictedData["leftLabels"])
predictedLabelsByPerceptronForLeftSplit = ProcessFeatureByPerceptronModel(labelIndicesOfPredictedData["leftExamples"], 
                                                                          labelIndicesOfPredictedData["leftLabels"])
print(predictedLabelsByPerceptronForLeftSplit)

