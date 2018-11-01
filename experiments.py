from DecisionTree.CompareAlgorithms import *
from DecisionTree.Tree import *
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.metrics import accuracy_score
from time import time

DEPTH = 3
NR_RAND_TREES = 200

X, y = load_breast_cancer(return_X_y=True)
print(y,"\n")

# BASIC TREE
print("\n------ BASIC TREE ------\n")
print("Depth: ", DEPTH)
t = Tree(max_depth=DEPTH)

st1 = time()
t.train(X, y)
en1 = time()
st1p = time()
y_pred = t.predict(X)
en1p = time()

basic_acc = accuracy_score(y,y_pred)
print(Get_ConfusionMatrix(y,y_pred))
print("F-Score: ", Get_F_Score(y,y_pred))
print("Accuracy: ", basic_acc)
print("Time to train:", en1-st1)
print("Time to test:", en1p-st1p)

if len(np.unique(y)) == 2:
    Generate_ROC_Curve(y,t.getClassProb(X))
# BASIC TREE END
# RANDOM TREES
print("\n------ RANDOM TREE ------\n")
print("Depth: ", DEPTH)
print("Nr trees: ", NR_RAND_TREES)
t2 = Tree(max_depth=DEPTH, random_feat=True)

t2_max = None
y2_pred = None
acc_max = 0

iterations_taken = NR_RAND_TREES
acc_list = []

st2 = time()
for i in range(0, NR_RAND_TREES):
    t2.train(X, y)
    y2_pred = t2.predict(X)
    acc = accuracy_score(y2_pred, y)
    acc_list.append(acc)
    if acc >= basic_acc:
        iterations_taken = i+1
        t2_max = t2
        acc_max = acc
        break
    if acc > acc_max:
        acc_max = acc
        t2_max = t2

en2 = time()
st2p = time()
y2_pred = t2_max.predict(X)
en2p = time()

print(Get_ConfusionMatrix(y,y2_pred))
print("F-Score: ", Get_F_Score(y,y2_pred))
print("Accuracy: ", acc_max)
print("Time to train:", en2-st2)
print("Iterations taken:", iterations_taken)
print("Time to test:", en2p-st2p)
random_decision_tree_accuracy(acc_list)

if len(np.unique(y)) == 2:
    Generate_ROC_Curve(y,t2_max.getClassProb(X))
# RANDOM TREES END
# LOOKAHEAD TREE
print("\n------ LOOKAHEAD TREE ------\n")
print("Depth: ", DEPTH)
t3 = Tree(max_depth=DEPTH, lookahead=True)

st3 = time()
t3.train(X, y)
en3 = time()
print("yo")
st3p = time()
y3_pred = t3.predict(X)
en3p = time()
print("yu")

basic_acc = accuracy_score(y,y3_pred)
print(Get_ConfusionMatrix(y,y3_pred))
print("F-Score: ", Get_F_Score(y,y3_pred))
print("Accuracy: ", accuracy_score(y,y3_pred))
print("Time to train:", en3-st3)
print("Time to test:", en3p-st3p)

if len(np.unique(y)) == 2:
    Generate_ROC_Curve(y,t3.getClassProb(X))

# LOOKAHEAD TREE END
