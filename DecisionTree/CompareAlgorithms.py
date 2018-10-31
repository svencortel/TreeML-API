from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics

def Get_F_Score(trueLabels, predictedLabel):
    return f1_score(trueLabels, predictedLabel, average='macro')

def Get_ConfusionMatrix(trueLabels, predictedLabel):
    return confusion_matrix(trueLabels, predictedLabel)

def Generate_ROC_Curve(trueLabels, probabiltyEstimateOfLabel):
    fpr, tpr, _ = metrics.roc_curve(trueLabels,  probabiltyEstimateOfLabel)
    auc = metrics.roc_auc_score(y_train, y_pred_proba)
    plt.plot(fpr,tpr,label="ROC curve area = "+str(auc))
    plt.xlabel("False Positve Rate")
    plt.ylabel("True Positve Rate")
    plt.title("ROC Curve")
    plt.legend(loc=4)
    plt.show() 
