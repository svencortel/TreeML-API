from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics
import matplotlib.pyplot as plt

def Get_F_Score(trueLabels, predictedLabel):
    return f1_score(trueLabels, predictedLabel, average='macro')

def Get_ConfusionMatrix(trueLabels, predictedLabel):
    return confusion_matrix(trueLabels, predictedLabel)

def Generate_ROC_Curve(trueLabels, probabiltyEstimateOfLabel):
    fpr, tpr, _ = metrics.roc_curve(trueLabels,  probabiltyEstimateOfLabel)
    auc = metrics.roc_auc_score(trueLabels, probabiltyEstimateOfLabel)
    f = plt.figure(1)
    plt.plot(fpr,tpr,label="ROC curve area = "+str(auc))
    plt.xlabel("False Positve Rate")
    plt.ylabel("True Positve Rate")
    plt.title("ROC Curve")
    f.legend(loc=4)
    f.show()
    
def random_decision_tree_accuracy(treesAccuracyList):
    num_bins = 20
    n, bins, patches = plt.hist(treesAccuracyList, num_bins, facecolor='blue', alpha=0.5)
    f = plt.figure(2)
    plt.xlabel('Accuracy')
    plt.ylabel('Number Of Trees')
    f.show()
