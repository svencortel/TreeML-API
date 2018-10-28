from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict
import numpy as np

class Node:
    def __init__(self, feature_index = None, IG_score = None, current_depth = 0, threshold = None):
        self.feature_index = feature_index
        self.IG_score = IG_score
        self.threshold = threshold
        self.depth = current_depth
        self.child_left = None
        self.child_right = None
        self.class_value = None

    def setLeftChild(self, node):
        self.child_left = node

    def setRightChild(self, node):
        self.child_right = node

    def isLeaf(self):
        return (self.child_right is None and self.child_left is None)

    def setClass(self, class_value):
        self.class_value = class_value

    def _getMajorityClass(self, y_data):
        (vals, counts) = np.unique(y_data, return_counts=True)
        return vals[np.argmax(counts)]

    def train(self, X_data, y_data, max_depth):
        if self.depth == max_depth:
            self.setClass(self._getMajorityClass(y_data))
        elif all(y_data[number] == y_data[0] for number in range(1, len(y_data) - 1)):
            self.setClass(y_data[0])
        else:
            features_IG = mutual_info_classif(X_data, y_data)
            self.IG_score = max(features_IG)
            # hopefully this doesn't return an array
            self.feature_index = features_IG.argmax()
            self.threshold = getNodeSplitLabels(X_data[:, self.feature_index],
                                                y_data)
            split_data = SplitMasterDataByThreshold(X_data, y_data, self.threshold,
                                                    self.feature_index)
            self.child_left = Node(current_depth=self.depth + 1)
            self.child_right = Node(current_depth=self.depth + 1)
            self.child_left.train(split_data['leftExamples'], split_data['leftLabels'],
                                  max_depth=max_depth)
            self.child_right.train(split_data['rightExamples'], split_data['rightLabels'],
                                   max_depth=max_depth)

    def predictData(self, data):
        if self.isLeaf():
            if self.class_value is None:
                raise("A class value has not been set to leaf node")
            return self.class_value

        if data[self.feature_index] <= self.threshold:
            return self.child_left.predictData(data)
        else:
            return self.child_right.predictData(data)

    def printNode(self):
        if not self.isLeaf():
            print("feature:", self.feature_index, "threshold:", self.threshold, "IG:", self.IG_score)
        else:
            print("leaf w class:", self.class_value)
            return

        self.child_left.printNode()
        self.child_right.printNode()

# Process each feature using the decision stump model
# Input: feature data, labels
# Return: Predicted value by the model

def getNodeSplitLabels(samples, labels):
    clf = DecisionTreeClassifier(max_depth=1, criterion="entropy")
    samples = samples.reshape(-1, 1)
    clf.fit(samples, labels)
    print(clf.score(samples, labels))
    return clf.tree_.threshold[0]

# Split master data by predicted labels by the model
# Input: predicted labels
# Return: Dictionary with the left and right sub-sets (split the master data)

def SplitMasterDataByThreshold(X_data, y_data, threshold, feature_index):

    return FilterData(X_data, y_data, threshold, feature_index)

# Get instances for the list of row indexes
# Input: sample data, list of row index
# Return: data instances at the provided index

def FilterData(x_data, y_data, threshold, feature_index):
    x_res_left = np.vstack(x_data[x_data[:,feature_index] <= threshold])
    y_res_left = np.vstack(y_data[x_data[:,feature_index] <= threshold])
    x_res_right = np.vstack(x_data[x_data[:,feature_index] > threshold])
    y_res_right = np.vstack(y_data[x_data[:,feature_index] > threshold])

    return {"leftExamples":x_res_left, "leftLabels":y_res_left,
            "rightExamples":x_res_right, "rightLabels":y_res_right}
