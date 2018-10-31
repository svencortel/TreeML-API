from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def Get_F_Score(trueLabels, predictedLabel):
    return f1_score(trueLabels, predictedLabel, average='macro')

def Get_ConfusionMatrix(trueLabels, predictedLabel):
    return confusion_matrix(trueLabels, predictedLabel)

