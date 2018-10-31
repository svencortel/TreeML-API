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

t = Tree(max_depth=DEPTH, PFSRT=True, omega=1.5)

t.train(X,y)
t.printTree()

t.updatePFSRT()

#print(t.DS)
#print(t.PS)
acc = []
for i in range(0,NR_TREES):
    t.train() # train with same data
    t.updatePFSRT()
    acc.append((i, t._cur_accuracy))

acc = sorted(acc, key=lambda kv: kv[1])
for tup in acc:
    print("Tree " + str(tup[0]) + " acc:", tup[1])

