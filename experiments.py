from DecisionTree.CompareAlgorithms import *
from DecisionTree.Tree import *
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.metrics import accuracy_score
from time import time
from statistics import stdev

def do_experiments(X, y, depth, nr_rand_trees, data_label):
    # BASIC TREE
    print("\n------ BASIC TREE ------\n")
    print("Depth: ", depth)
    t = Tree(max_depth=depth)

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

    # this is 10 fold cross validation
    cv_arr = cross_validate(t, X, y, cv=10)
    print("Accuracy after 10-fold CV:", float(sum(cv_arr)) / max(len(cv_arr), 1),
          "(", stdev(cv_arr), ")")

    if len(np.unique(y)) == 2:
        Generate_ROC_Curve(y,t.getClassProb(X), "ID3", label_text=data_label)
    # BASIC TREE END
    # RANDOM TREES
    print("\n------ RANDOM TREE ------\n")
    print("Depth: ", depth)
    print("Nr trees: ", nr_rand_trees)
    t2 = Tree(max_depth=depth, random_feat=True)

    t2_max = None
    y2_pred = None
    acc_max = 0

    iterations_taken = nr_rand_trees
    acc_list = []
    max_accs = []

    st2 = time()
    for i in range(0, nr_rand_trees):
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
        max_accs.append(acc_max)

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


    # this is 10 fold cross validation
    cv_arr = cross_validate(t2_max, X, y, cv=10)
    print("Accuracy after 10-fold CV:",float(sum(cv_arr)) / max(len(cv_arr), 1),
          "(",stdev(cv_arr),")")

    random_decision_tree_accuracy(acc_list, label_text=data_label)
    accuracyRiseForRandomTrees(max_accs, label_text=data_label)

    if len(np.unique(y)) == 2:
        Generate_ROC_Curve(y,t2_max.getClassProb(X), "Random Forest",
                           label_text=data_label)
    # RANDOM TREES END
    # LOOKAHEAD TREE
    print("\n------ LOOKAHEAD TREE ------\n")
    print("Depth: ", depth)
    t3 = Tree(max_depth=depth, lookahead=True)

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

    # this is 10 fold cross validation
    cv_arr = cross_validate(t3, X, y, cv=5)
    print("Accuracy after 10-fold CV:", float(sum(cv_arr)) / max(len(cv_arr), 1),
          "(", stdev(cv_arr), ")")

    if len(np.unique(y)) == 2:
        Generate_ROC_Curve(y, t3.getClassProb(X), "Lookahead DT",
                           label_text=data_label)

    # LOOKAHEAD TREE END


    # print("\n")
    # t.printTree()
    # print("\n")
    # t2_max.printTree()
    # print("\n")
    # t3.printTree()

X, y = load_breast_cancer(return_X_y=True)
do_experiments(X, y, 3, 5000, "Breast Cancer")

X, y = load_iris(return_X_y=True)
do_experiments(X, y, 3, 5000, "Iris")

X, y = load_wine(return_X_y=True)
do_experiments(X, y, 3, 5000, "Wine")

X, y = load_digits(return_X_y=True)
do_experiments(X, y, 6, 5000, "Digits")
