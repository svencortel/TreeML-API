from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from random import randint, seed, random
from math import log2

class Node:
    def __init__(self, feature_index = None, IG_score = None, current_depth = 0,
                 threshold = None, random_feat = False, tree = None, parent = None):
        self.feature_index = feature_index
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
        if self.depth == max_depth:
            self.setClass(self._getMajorityClass(y_data))
            self._setPosProb(y_data)
        elif all(y_data[number] == y_data[0] for number in range(1, len(y_data) - 1)):
            self.setClass(y_data[0][0])
            self._setPosProb(y_data)
        elif self.depth < max_depth - 1:
            #FEATURE AND THRESHOLD SELECTION THROUGH LOOKAHEAD
            self.feature_index, self.threshold, self.IG_score = self.lookahead(X_data,
                                                                               y_data)
            if self.threshold is None:
                self.setClass(self._getMajorityClass(y_data))
                self._setPosProb(y_data)
                return

            split_data = FilterData(X_data, y_data, self.threshold, self.feature_index)
            self.child_left = Node(current_depth=self.depth + 1,
                                   tree=self._tree, parent=self)
            self.child_right = Node(current_depth=self.depth + 1,
                                    tree=self._tree, parent=self)
            self.child_left.train_lookahead(split_data['leftExamples'],
                                            split_data['leftLabels'],
                                            max_depth=max_depth)
            self.child_right.train_lookahead(split_data['rightExamples'],
                                             split_data['rightLabels'],
                                             max_depth=max_depth)
        else:
            print("ey")
            # if next node is leaf then train the node normally
            self.train(X_data, y_data, max_depth)

    def lookahead(self, X_data, y_data):
        best_IG=0
        best_feat=0
        best_threshold = None

        for feature in range(0, X_data.shape[1]):
            if all(X_data[:, feature] == X_data[0, feature]):
                continue
            data = np.unique(sorted(X_data[:,feature]))
            #print(data)

            # calculate entropy of this node
            cur_entropy = 0
            (vals, counts) = np.unique(y_data, return_counts=True)
            data_len = X_data.shape[0]
            for count in counts:
                cur_entropy += -(count/data_len * log2(count/data_len))
            #print(cur_entropy)

            #print(data)
            for i in range(0, len(data) - 2):
                t = (data[i] + data[i+1]) / 2
                split = FilterData(X_data,y_data,t,feature)
                child_left = Node(tree=self._tree)
                child_right = Node(tree=self._tree)
                #print(split["leftExamples"][:,feature])
                #print(split["rightExamples"][:, feature])

                ig1 = child_left.getIG(split["leftExamples"], split["leftLabels"])
                if ig1 is None:
                    ig1 = cur_entropy

                ig2 = child_right.getIG(split["rightExamples"], split["rightLabels"])
                if ig2 is None:
                    ig2 = cur_entropy

                cur_ig = (ig1 * split["leftExamples"].shape[0] +\
                          ig2 * split["rightExamples"].shape[0]) / data_len

                if cur_ig > best_IG:
                    best_IG = cur_ig
                    best_threshold = t
                    best_feat = feature

        return (best_feat, best_threshold, best_IG)

    def getIG(self, X_data, y_data):
        self.selectFeatureByScore2(X_data, y_data)
        if self.feature_index is None:
            return 0
        return self.IG_score

    def train(self, X_data, y_data, max_depth):
        if self.depth == max_depth:
            self.setClass(self._getMajorityClass(y_data))
            self._setPosProb(y_data)
        elif all(y_data[number] == y_data[0] for number in range(1, len(y_data) - 1)):
            self.setClass(y_data[0][0])
            self._setPosProb(y_data)
        else:
            # FEATURE SELECTION
            self.selectFeature(X_data, y_data)
            if self.feature_index is None:
                self.setClass(self._getMajorityClass(y_data))
                self._setPosProb(y_data)
                return
            # FEATURE SELECTION END

            #THRESHOLD SELECTION
            #self.selectThreshold(X_data, y_data, self.feature_index)
            #if self.threshold is None and self.class_value is not None:
            #    return
            #elif self.threshold is None and self.class_value is None:
            #    raise Exception("Node is neither leaf nor splitting")
            #THRESHOLD SELECTION END

            split_data = FilterData(X_data, y_data, self.threshold, self.feature_index)
            self.child_left = Node(current_depth=self.depth + 1, random_feat=self.random,
                                   tree=self._tree, parent=self)
            self.child_right = Node(current_depth=self.depth + 1, random_feat=self.random,
                                    tree=self._tree, parent=self)
            self.child_left.train(split_data['leftExamples'], split_data['leftLabels'],
                                  max_depth=max_depth)
            self.child_right.train(split_data['rightExamples'], split_data['rightLabels'],
                                   max_depth=max_depth)

    def selectFeature(self, X_data, y_data):
        if self.random:
            self.selectFeatureByRandom(X_data, y_data)
        elif self._tree.is_PFSRT:
            self.selectFeatureByPFSRT(X_data, y_data)
        else:
            self.selectFeatureByScore2(X_data, y_data)

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
        #print(self.threshold)

    def selectFeatureByScore2(self, X_data, y_data, criterion="entropy"):
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=1)
        clf.fit(X_data, y_data)

        self.feature_index = clf.tree_.feature[0]
        self.threshold = clf.tree_.threshold[0]
        N_left = len(y_data[X_data[:, self.feature_index] <= self.threshold])
        N_right = len(y_data[X_data[:, self.feature_index] > self.threshold])
        N_t = X_data.shape[0]
        if clf.tree_.impurity[0] == 0:
            self.IG_score = None
            return

        self.IG_score = X_data.shape[0] / self._tree._nr_examples * (clf.tree_.impurity[0]
                                                                     - N_left /N_t * clf.tree_.impurity[1]
                                                                     - N_right/N_t * clf.tree_.impurity[2])


    def selectFeatureByScore(self, X_data, y_data, criterion="entropy"):

        feature_index_association = {k: k for k in range(0, X_data.shape[1])}
        # pr = False
        # find all features that have all their data equal
        feature = 0
        while feature < X_data.shape[1]:
            if all(X_data[:, feature] == X_data[0, feature]):
                # print("Found useless feature: ", feature)
                # if not pr:
                #     print("X_data before: ", X_data)
                #     pr = True
                # print(feature_index_association)
                for feat in range(feature, len(feature_index_association) - 2):
                    feature_index_association[feat] = feature_index_association[feat + 1]
                del feature_index_association[len(feature_index_association) - 1]
                # print(feature_index_association)
                X_data = np.delete(X_data, feature, axis=1)
                feature -= 1
            feature += 1

        if feature_index_association == {}:
            self.feature_index = None
            return
        # if(pr):
        #     print("X_data after: ", X_data)
        # clf = DecisionTreeClassifier(max_depth=1, criterion="entropy")
        # clf.fit(X_data, y_data)

        features_IG = mutual_info_classif(X_data, y_data.ravel())
        self.IG_score = max(features_IG)
        # hopefully this doesn't return an array
        #print(feature_index_association)
        self.feature_index = feature_index_association[features_IG.argmax()]

    def selectFeatureByRandom(self, X_data, y_data):
        self.IG_score = -1 # -1 for random criteria selection
        feat_assoc = self._uselessFeatureElimination(X_data)
        if feat_assoc == {}:
            self.feature_index = None
            return
        naive_feat = randint(0, len(feat_assoc)-1)
        self.feature_index = feat_assoc[naive_feat]
        self.threshold = getNodeSplitThreshold(X_data[:, self.feature_index], y_data)

    def selectFeatureByPFSRT(self, X_data, y_data):
        self.IG_score = -1
        assoc_dict = self._uselessFeatureElimination(X_data)
        if assoc_dict == {}:
            self.feature_index = None
            return
        self.feature_index = self._ProbabilisticFeatureSelection(assoc_dict)
        self.threshold = getNodeSplitThreshold(X_data[:, self.feature_index], y_data)

    def _uselessFeatureElimination(self, X_data):
        feature_index_association = {k: k for k in range(0, X_data.shape[1])}
        # pr = False
        # find all features that have all their data equal
        feature = 0
        while feature < X_data.shape[1]:
            if all(X_data[:, feature] == X_data[0, feature]):
                # print("Found useless feature: ", feature)
                # if not pr:
                #     print("X_data before: ", X_data)
                #     pr = True
                # print(feature_index_association)
                for feat in range(feature, len(feature_index_association) - 2):
                    feature_index_association[feat] = feature_index_association[feat + 1]
                del feature_index_association[len(feature_index_association) - 1]
                # print(feature_index_association)
                X_data = np.delete(X_data, feature, axis=1)
                feature -= 1
            feature += 1
        return feature_index_association

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
                return assoc_dict[i]
        return assoc_dict[len(assoc_dict)-1]

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

