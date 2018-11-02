from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from random import randint, seed, random
from math import log2

class Node:
    def __init__(self, feature_index = None, IG_score = None, current_depth = 0,
                 threshold = None, random_feat = False, tree = None, parent = None,
                 feature_assoc = None):
        self.feature_index = feature_index
        self.feature_index_key = None
        self.IG_score = IG_score
        self.threshold = threshold
        self.depth = current_depth
        self.random = random_feat
        self._tree = tree
        self.child_left = None
        self.child_right = None
        self.parent = parent
        self.class_value = None
        self.posProb = None
        self.feature_index_key = None
        self.feature_assoc = feature_assoc
        seed()

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

    def _setPosProb(self, y_data):
        if self._tree.isBinaryClassifier():
            nr_pos = 0
            nr_neg = 0
            for i in y_data:
                if i > 0:
                    nr_pos += 1
                else:
                    nr_neg += 1

            self.posProb = nr_pos/(nr_neg+nr_pos)

    def train_lookahead(self, X_data, y_data, max_depth):
        if self.feature_assoc is None:
            self.feature_assoc = {k: k for k in range(0, X_data.shape[1])}
        if self.depth == max_depth:
            self.setClass(self._getMajorityClass(y_data))
            self._setPosProb(y_data)
        elif all(y_data[number] == y_data[0] for number in range(1, len(y_data) - 1)):
            self.setClass(y_data[0][0])
            self._setPosProb(y_data)
        elif self.depth < max_depth - 1:
            #FEATURE AND THRESHOLD SELECTION THROUGH LOOKAHEAD
            X_data = self._uselessFeatureElimination(X_data)
            if self.feature_assoc == {}:
                self.setClass(self._getMajorityClass(y_data))
                self._setPosProb(y_data)
                return

            self.feature_index, self.threshold, self.IG_score = self.lookahead(X_data,
                                                                               y_data)
            if self.threshold is None:
                self.setClass(self._getMajorityClass(y_data))
                self._setPosProb(y_data)
                return

            split_data = FilterData(X_data, y_data, self.threshold,
                                    self.feature_index_key)
            self.child_left = Node(current_depth=self.depth + 1,
                                   tree=self._tree, parent=self,
                                   feature_assoc=dict(self.feature_assoc))
            self.child_right = Node(current_depth=self.depth + 1,
                                    tree=self._tree, parent=self,
                                    feature_assoc=dict(self.feature_assoc))
            self.child_left.train_lookahead(split_data['leftExamples'],
                                            split_data['leftLabels'],
                                            max_depth=max_depth)
            self.child_right.train_lookahead(split_data['rightExamples'],
                                             split_data['rightLabels'],
                                             max_depth=max_depth)
        else:
            # if next node is leaf then train the node normally
            self.train(X_data, y_data, max_depth)

    def lookahead(self, X_data, y_data):
        best_IG=0
        best_feat=0
        best_threshold = None

        # calculate entropy of this node
        cur_entropy = 0
        (_, counts) = np.unique(y_data, return_counts=True)
        data_len = X_data.shape[0]
        for count in counts:
            cur_entropy += -(count / X_data.shape[0] * log2(count / X_data.shape[0]))

        for feature in range(0, X_data.shape[1]):
            data = np.unique(sorted(X_data[:, feature]))

            # set theshold between each two data values and lookahead for potential Information Gain
            for i in range(0, len(data) - 2):
                t = (data[i] + data[i+1]) / 2
                split = FilterData(X_data,y_data,t,feature)
                child_left = Node(tree=self._tree,
                                  feature_assoc=dict(self.feature_assoc))
                child_right = Node(tree=self._tree,
                                   feature_assoc=dict(self.feature_assoc))

                ig1 = child_left.getIG(split["leftExamples"], split["leftLabels"])
                if ig1 is None:
                    ig1 = cur_entropy

                ig2 = child_right.getIG(split["rightExamples"], split["rightLabels"])
                if ig2 is None:
                    ig2 = cur_entropy

                cur_ig = (ig1 * split["leftExamples"].shape[0] +
                          ig2 * split["rightExamples"].shape[0]) / X_data.shape[0]

                if cur_ig > best_IG:
                    best_IG = cur_ig
                    best_threshold = t
                    best_feat = feature

        self.feature_index_key = best_feat
        return (self.feature_assoc[best_feat], best_threshold, best_IG)

    def getIG(self, X_data, y_data):
        self.selectFeatureByScore2(X_data, y_data)
        if self.feature_index is None:
            return 0
        return self.IG_score

    def train(self, X_data, y_data, max_depth):
        if self.feature_assoc is None:
            self.feature_assoc = {k: k for k in range(0, X_data.shape[1])}
        if self.depth == max_depth:
            self.setClass(self._getMajorityClass(y_data))
            self._setPosProb(y_data)
        elif all(y_data[number] == y_data[0] for number in range(1, len(y_data) - 1)):
            self.setClass(y_data[0][0])
            self._setPosProb(y_data)
        else:
            # FEATURE AND THRESHOLD SELECTION
            if X_data.shape[1] != len(self.feature_assoc):
                raise Exception("X_data columns not equal to length of feature assoc BEFORE ANYTHING:\n",
                                self.feature_assoc, "\n", X_data.shape)
            X_data = self.selectFeature(X_data, y_data)

            if self.feature_index is None:
                self.setClass(self._getMajorityClass(y_data))
                self._setPosProb(y_data)
                return
            # FEATURE AND THRESHOLD SELECTION END

            # split data and train children
            split_data = FilterData(X_data, y_data, self.threshold,
                                    self.feature_index_key)
            if split_data["leftExamples"].shape[1] != len(self.feature_assoc):
                raise Exception("left ex columns not equal to length of feature assoc right after split:\n",
                                self.feature_assoc, "\n", split_data["leftExamples"].shape)

            if split_data["rightExamples"].shape[1] != len(self.feature_assoc):
                raise Exception("right ex columns not equal to length of feature assoc right after split:\n",
                                self.feature_assoc, "\n", split_data["rightExamples"].shape)
            # print(X_data.shape)
            # print(split_data["leftExamples"].shape)
            # print(split_data["rightExamples"].shape)
            # print(len(self.feature_assoc))
            # print("\n")


            self.child_left = Node(current_depth=self.depth + 1, random_feat=self.random,
                                   tree=self._tree, parent=self,
                                   feature_assoc=dict(self.feature_assoc))
            self.child_right = Node(current_depth=self.depth + 1, random_feat=self.random,
                                    tree=self._tree, parent=self,
                                    feature_assoc=dict(self.feature_assoc))
            self.child_left.train(split_data['leftExamples'], split_data['leftLabels'],
                                  max_depth=max_depth)
            self.child_right.train(split_data['rightExamples'], split_data['rightLabels'],
                                   max_depth=max_depth)

    def selectFeature(self, X_data, y_data):
        if self.random:
            return self.selectFeatureByRandom(X_data, y_data)
        elif self._tree.is_PFSRT:
            return self.selectFeatureByPFSRT(X_data, y_data)
        else:
            return self.selectFeatureByScore2(X_data, y_data)

    def selectThreshold(self, X_data, y_data, feature_index):
        # if all X data for this feature are equal, then we can't split
        if all(X_data[:, self.feature_index] == X_data[0, self.feature_index]):
            if not self._tree.is_PFSRT:
                raise Exception("Error in choosing feature to split by, "
                                "maybe labels are continuous?")
            self.setClass(self._getMajorityClass(y_data))
            self._setPosProb(y_data)
            return

        # if we are here then a proper split can be made
        self.threshold = getNodeSplitThreshold(X_data[:, self.feature_index], y_data)

    def selectFeatureByScore2(self, X_data, y_data, criterion="entropy"):
        X_data = self._uselessFeatureElimination(X_data)
        #print(X_data.shape)
        if self.feature_assoc == {}:
            self.feature_index = None
            # no need to return anything because this is a leaf node
            return

        clf = DecisionTreeClassifier(criterion=criterion, max_depth=1)
        clf.fit(X_data, y_data)
        if clf.tree_.impurity[0] == 0:
            self.IG_score = None
            self.feature_index = None
            self.threshold = None
            return X_data
        self.feature_index_key = clf.tree_.feature[0]
        #print(self.feature_assoc)
        self.feature_index = self.feature_assoc[self.feature_index_key]
        self.threshold = clf.tree_.threshold[0]
        N_left = len(y_data[X_data[:, self.feature_index_key] <= self.threshold])
        N_right = len(y_data[X_data[:, self.feature_index_key] > self.threshold])
        N_t = X_data.shape[0]

        self.IG_score = X_data.shape[0] / self._tree._nr_examples * (clf.tree_.impurity[0]
                                                                     - N_left /N_t * clf.tree_.impurity[1]
                                                                     - N_right/N_t * clf.tree_.impurity[2])
        return X_data

    # def selectFeatureByScore(self, X_data, y_data, criterion="entropy"):
    #     X_data = self._uselessFeatureElimination(X_data)
    #     features_IG = mutual_info_classif(X_data, y_data.ravel())
    #     self.IG_score = max(features_IG)
    #     self.feature_index = self.feature_assoc[features_IG.argmax()]
    #     # return feature reduced data
    #     return X_data

    def selectFeatureByRandom(self, X_data, y_data):
        self.IG_score = -1 # -1 for random criteria selection
        X_data = self._uselessFeatureElimination(X_data)
        if self.feature_assoc == {}:
            self.feature_index = None
            # no need to return anything because this is a leaf node
            return
        self.feature_index_key = randint(0, len(self.feature_assoc)-1)
        self.feature_index = self.feature_assoc[self.feature_index_key]
        self.threshold = getNodeSplitThreshold(X_data[:, self.feature_index_key], y_data)
        # get the feature reduced data
        return X_data

    def selectFeatureByPFSRT(self, X_data, y_data):
        self.IG_score = -1
        X_data = self._uselessFeatureElimination(X_data)
        if self.feature_assoc == {}:
            self.feature_index = None
            # no need to return anything because this is a leaf node
            return
        self.feature_index, self.feature_index_key = self._ProbabilisticFeatureSelection(self.feature_assoc)
        # key_feature_index because the X_data is modified
        self.threshold = getNodeSplitThreshold(X_data[:, self.feature_index_key], y_data)
        # get the feature reduced data
        return X_data

    # returns (dictionary, modified_data) pair
    def _uselessFeatureElimination(self, X_data):
        if len(self.feature_assoc) != X_data.shape[1]:
            raise Exception("X_data columns not equal to length of feature assoc BEFORE:\n",
                            self.feature_assoc, "\n", X_data.shape)
        # feature_index_association = {k: k for k in range(0, X_data.shape[1])}
        # pr = False
        # find all features that have all their data equal
        feature = 0
        while feature < X_data.shape[1]:
            if all(X_data[:, feature] == X_data[0, feature]):
                for feat in range(feature, len(self.feature_assoc) - 2):
                    self.feature_assoc[feat] = self.feature_assoc[feat + 1]
                del self.feature_assoc[len(self.feature_assoc) - 1]

                X_data = np.delete(X_data, feature, axis=1)
                feature -= 1
            feature += 1

            if len(self.feature_assoc) != X_data.shape[1]:
                raise Exception("X_data columns not equal to length of feature assoc AFTER:\n",
                                self.feature_assoc, "\n", X_data.shape)
        return X_data

    # returns (value, key) pair
    def _ProbabilisticFeatureSelection(self, assoc_dict):
        # magnitudes array
        M = []
        for i in range(0, len(assoc_dict)-1):
            M.append(self._tree.DS[assoc_dict[i]][self.depth] *  # + or * ?
                     (self._tree.PS[assoc_dict[i]][self.parent.feature_index]
                     if self.parent is not None else
                      self._tree.PS[assoc_dict[i]][self._tree.nr_features]))

        randv = random() * sum(M)

        for i in range(0, len(assoc_dict)-1):
            if randv < sum(M[:i]):
                return assoc_dict[i], i
        return assoc_dict[len(assoc_dict)-1], len(assoc_dict) - 1

    def predictData(self, data):
        if self.isLeaf():
            if self.class_value is None:
                raise("A class value has not been set to leaf node")
            return self.class_value

        if data[self.feature_index] <= self.threshold:
            return self.child_left.predictData(data)
        else:
            return self.child_right.predictData(data)

    def getPositiveProb(self, data):
        if self.isLeaf():
            return self.posProb

        if data[self.feature_index] <= self.threshold:
            return self.child_left.getPositiveProb(data)
        else:
            return self.child_right.getPositiveProb(data)

    def printNode(self):
        if not self.isLeaf():
            print("feature:", self.feature_index, "threshold:", self.threshold, "IG:", self.IG_score)
        else:
            print("leaf w class:", self.class_value)
            return

        self.child_left.printNode()
        self.child_right.printNode()

    def recursiveUpdatePFSRT(self):
        if not self.isLeaf():
            var = self._tree.DS[self.feature_index][self.depth]
            modifier = self._tree.theta
            parent_feature = (self.parent.feature_index if self.parent is not None else
                              self._tree.nr_features)
            if self._tree._cur_accuracy > self._tree._best_accuracy:
                modifier = self._tree.omega
            self._tree.DS[self.feature_index][self.depth] = (var +
                (self._tree._cur_accuracy - self._tree._best_accuracy) * var) * modifier
            self._tree.PS[self.feature_index][parent_feature] = (var +
                (self._tree._cur_accuracy - self._tree._best_accuracy) * var) * modifier

            self.child_left.recursiveUpdatePFSRT()
            self.child_right.recursiveUpdatePFSRT()

    def getClassProb(self, data):
        if self.isLeaf():
            return

# Process each feature using the decision stump model
# Input: feature data, labels
# Return: Predicted value by the model
def getNodeSplitThreshold(samples, labels):
    clf = DecisionTreeClassifier(max_depth=1, criterion="entropy")
    samples = samples.reshape(-1, 1)
    clf.fit(samples, labels)
    return clf.tree_.threshold[0]

# Split X data by threshold and feature
# Input: predicted labels
# Return: Dictionary with the left and right sub-sets (split the master data)
def FilterData(x_data, y_data, threshold, feature_index):
    x_res_left = np.vstack(x_data[x_data[:,feature_index] <= threshold])
    y_res_left = np.vstack(y_data[x_data[:,feature_index] <= threshold])
    x_res_right = np.vstack(x_data[x_data[:,feature_index] > threshold])
    y_res_right = np.vstack(y_data[x_data[:,feature_index] > threshold])

    return {"leftExamples":x_res_left, "leftLabels":y_res_left,
            "rightExamples":x_res_right, "rightLabels":y_res_right}

