from DecisionTree.Tree import *
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from time import time

DEPTH = 2

iris=load_iris()
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

print(X)
st1 = time()
t=Tree(max_depth=DEPTH, lookahead=True)
t.train(X, y)
en1 = time()

t.printTree()

print(y)
st1p = time()
y_pred=t.predict(X)
en1p = time()
print(y_pred)

print(getAccuracy(y, y_pred))

clf = DecisionTreeClassifier(criterion='entropy', max_depth=DEPTH)
st2 = time()
clf.fit(X, y)
en2 = time()
st2p = time()
y_pred2 = clf.predict(X)
en2p = time()
#print(y_pred2)
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
