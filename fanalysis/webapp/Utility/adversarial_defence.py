import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix

class AdversarialDefence:

    def __init__(self, xTrain, xTrainAdv, xTestAdv, yTrain,yTest, model):
        self.xTrain = xTrain
        self.xTrainAdv = xTrainAdv
        self.yTrain = yTrain
        self.xTestAdv = xTestAdv
        self.yTest = yTest
        self.model = model
        self.dataAmount = 100

    def defence(self):

        new_x_train = np.append(self.xTrain[:self.dataAmount], self.xTrainAdv, axis=0)
        new_y_train = np.append(self.yTrain[:self.dataAmount], self.yTrain[:self.dataAmount], axis=0)

        self.model.fit(new_x_train, new_y_train)
        new_yPred = self.model.predict(self.xTestAdv)
        acc = accuracy_score(self.yTest[:self.dataAmount], new_yPred)

        yPred = self.model.predict(self.xTestAdv)

        prec = precision_score(self.yTest[:self.dataAmount], yPred)
        print("The precision is {}".format(prec))

        rec = recall_score(self.yTest[:self.dataAmount], yPred)
        print("The recall is {}".format(rec))

        f1 = f1_score(self.yTest[:self.dataAmount], yPred)
        print("The F1-Score is {}".format(f1))

        y_pred_proba = self.model.predict_proba(self.xTestAdv)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(self.yTest[:self.dataAmount], y_pred_proba)

        plt.figure(figsize=(4, 4))
        plt.plot([0, 1], [0, 1], linestyle='--')
        # plot the roc curve for the model
        plt.plot(fpr, tpr, marker='.', label='ROC Curve')
        plt.title('ROC Curve')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('media/ad_env_roc.png', dpi=100)
        plt.legend()

        LABELS = ['Normal', 'Fraud']
        conf_matrix = confusion_matrix(self.yTest[:self.dataAmount], yPred)
        plt.figure(figsize=(4, 4))
        sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
        plt.title("Confusion matrix")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.savefig('media/ad_env_confusion_matrix.png', dpi=100)

        return acc, prec, rec, f1