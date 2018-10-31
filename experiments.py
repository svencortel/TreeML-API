from DecisionTree.CompareAlgorithms import *
from DecisionTree.Tree import *
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from time import time

DEPTH = 5
NR_RAND_TREES = 2000

X, y = load_digits(return_X_y=True)
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

st2 = time()
for i in range(0, NR_RAND_TREES):
    t2.train(X, y)
    y2_pred = t2.predict(X)
    acc = accuracy_score(y2_pred, y)
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

if len(np.unique(y)) == 2:
    Generate_ROC_Curve(y,t2_max.getClassProb(X))
# RANDOM TREES END
# LOOKAHEAD TREE
print("\n------ LOOKAHEAD TREE ------\n")


# LOOKAHEAD TREE END
