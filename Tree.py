from sklearn.tree import DecisionTreeClassifier
import numpy as np
from Node import *

class Tree:
    """The structure of the tree"""
    def __init__(self, criterion = "entropy", max_depth = None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.X_data = None
        self.y_data = None
        self.root_node = None

    def load(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def train(self, X_data = None, y_data = None):
        if X_data is None and y_data is None:
            if self.X_data is None:
                raise("No data loaded")
            X_data = self.X_data
            y_data = self.y_data

        # init root node
        self.root_node = Node()
        self.root_node.train(X_data, y_data, self.max_depth)

    def predict(self, X_data):
        result = []
        for i in X_data:
            result.append(self.root_node.predictData(i))
        return np.array(result)

    def printTree(self):
        self.root_node.printNode()

def getDataSplit(samples, labels):
    if all(labels[number] == labels[0] for number in range(1, len(labels) - 1)):
        return labels[0]
    else:
        print('label values:', labels)
        clf = DecisionTreeClassifier(max_depth=1, criterion="entropy")
        clf.fit(samples, labels)
        # cv = CountVectorizer(max_df=0.95, min_df=2,
        #                     max_features=10000, stop_words='english')
        print(clf.score(samples, labels))
        print("Tree data:", clf.tree_.threshold[0])
        return clf.predict(samples)

# In[9]:

# Get the feature data from a data-set for a provided feature index
# Input: feature data, feature index for which data needs to be extracted
# Return: Data in the given feature

# Modify this to use numpy builtin column selector
def GetFeatureData(samples, featureIndex):
    return samples[:,featureIndex]
    #feature_data = []
    #for idx, data in enumerate(samples):
    #    array = []
    #    array.append(samples[idx][featureIndex])
    #    feature_data.append(array)
    #return feature_data
