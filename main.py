from DecisionTree.Tree import *
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from time import time

NR_TREES = 500
DEPTH = 4

iris=load_digits()
X=iris.data
y=iris.target

def getAccuracy(pred_y, actu_y):
    if len(pred_y) != len(actu_y):
        raise("The fuck?")
    correct = 0
    for i in range(0, len(pred_y)):
        if pred_y[i] == actu_y[i]:
            correct += 1

    return correct/len(pred_y)

st1 = time()
t = []
# generate 10 random trees and look at the dist
for i in range(0,NR_TREES):
    t.append(Tree(max_depth=DEPTH, random_feat=True))
    t[i].train(X, y)
en1 = time()

for i in range(0,NR_TREES):
    print("Tree",i)
    t[i].printTree()

print(y)
st1p = time()
y_pred = []
for i in range(0,NR_TREES):
    y_pred.append(t[i].predict(X))
en1p = time()

acc_tuples = []
for i in range(0, NR_TREES):
    acc_tuples.append(( i, getAccuracy(y, y_pred[i]) ))

acc_tuples = sorted(acc_tuples, key=lambda kv: kv[1])

for tup in acc_tuples:
    print("Tree " + str(tup[0]) + " acc:", tup[1])

clf = DecisionTreeClassifier(criterion='entropy', max_depth=DEPTH)
st2 = time()
clf.fit(X, y)
en2 = time()
st2p = time()
y_pred2 = clf.predict(X)
en2p = time()
print(getAccuracy(y, y_pred2))

print("Custom alg. time to train:", en1-st1)
print("Custom alg. time to test:", en1p-st1p)
print("SKLearn alg. time to train:", en2-st2)
print("SKLearn alg. time to test:", en2p-st2p)

dot_data = export_graphviz(clf, out_file=None,
                            filled=True, rounded=True,
                            special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("treeGraph")
