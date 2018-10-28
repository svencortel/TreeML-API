from Tree import *
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import graphviz

iris=load_breast_cancer()
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

t = Tree(max_depth=2)
t.train(X, y)

t.printTree()

print(y)
y_pred = t.predict(X)
print(y_pred)
print(getAccuracy(y, y_pred))

clf = DecisionTreeClassifier(criterion='entropy', max_depth=2)
clf.fit(X, y)
y_pred2 = clf.predict(X)
print(y_pred2)
print(getAccuracy(y, y_pred2))

dot_data = export_graphviz(clf, out_file=None,
                            filled=True, rounded=True,
                            special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("treeGraph")
