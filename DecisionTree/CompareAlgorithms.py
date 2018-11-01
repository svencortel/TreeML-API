from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np

def Get_F_Score(trueLabels, predictedLabel):
    return f1_score(trueLabels, predictedLabel, average='macro')

def Get_ConfusionMatrix(trueLabels, predictedLabel):
    return confusion_matrix(trueLabels, predictedLabel)

def Generate_ROC_Curve(trueLabels, probabiltyEstimateOfLabel, alg_name, label_text=""):
    fpr, tpr, _ = metrics.roc_curve(trueLabels,  probabiltyEstimateOfLabel)
    auc = metrics.roc_auc_score(trueLabels, probabiltyEstimateOfLabel)
    f = plt.figure(1)
    ax = f.add_subplot(111)
    ax.plot(fpr,tpr,label=alg_name + " ROC curve area = "+str(auc)+" "+label_text)
    ax.set_xlabel("False Positve Rate")
    ax.set_ylabel("True Positve Rate")
    ax.set_title("ROC Curve")
    f.legend(loc=4)
    f.show()
    
def random_decision_tree_accuracy(treesAccuracyList, label_text=""):
    num_bins = 20
    f = plt.figure(2)
    ax = f.add_subplot(111)
    n, bins, patches = ax.hist(treesAccuracyList, num_bins, facecolor='blue', alpha=0.5,
                               label=label_text)
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Number Of Trees')
    ax.set_title("Distribution of Trees based on Training Accuracy")
    f.show()

def accuracyRiseForRandomTrees(max_accs, label_text=""):
    f = plt.figure(3)
    ax = f.add_subplot(111)
    ax.plot(np.arange(1,len(max_accs)+1), max_accs, label=label_text)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_title("Best Tree Accuracy Improvement in Random Forest")
    f.show()

def cross_validate(tree_obj, X, y, cv = 3):
    skf = StratifiedKFold(n_splits=cv, shuffle=True)
    skf.get_n_splits(X, y)

    output = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        tree_obj.train(X_train, y_train)
        acc = metrics.accuracy_score(y_test, tree_obj.predict(X_test))
        output.append(acc)

    return output