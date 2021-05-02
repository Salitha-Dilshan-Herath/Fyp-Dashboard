import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt_curve
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pickle


class AdversarialDefence:

    def __init__(self, xDataTrain, xTrainAdv, xTestAdv, yDataTest, xDataTest, model):
        self.xTrain = xDataTrain
        self.xTrainAdv = xTrainAdv
        self.yTrain = yDataTest
        self.xTestAdv = xTestAdv
        self.yTest = xDataTest
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
        print("precision value is {}".format(prec))

        rec = recall_score(self.yTest[:self.dataAmount], yPred)
        print("recall value is {}".format(rec))

        f1 = f1_score(self.yTest[:self.dataAmount], yPred)
        print("F1-Score value is {}".format(f1))

        y_pred_proba = self.model.predict_proba(self.xTestAdv)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(self.yTest[:self.dataAmount], y_pred_proba)

        plt_curve.figure(figsize=(4, 4))
        plt_curve.plot([0, 1], [0, 1], linestyle='--')
        # genarate the roc_curve for the model
        plt_curve.plot(fpr, tpr, marker='.', label='ROC Curve')
        plt_curve.title('ROC Curve')
        plt_curve.ylabel('True Positive Rate')
        plt_curve.xlabel('False Positive Rate')
        plt_curve.savefig('media/ad_env_roc.png', dpi=100)
        plt_curve.legend()

        axiesLables = ['Normal', 'Fraud']
        conf_matrix = confusion_matrix(self.yTest[:self.dataAmount], yPred)
        plt_curve.figure(figsize=(4, 4))
        sns.heatmap(conf_matrix, xticklabels=axiesLables, yticklabels=axiesLables, annot=True, fmt="d");
        plt_curve.title("Confusion matrix")
        plt_curve.ylabel('True class')
        plt_curve.xlabel('Predicted class')
        plt_curve.savefig('media/ad_env_confusion_matrix.png', dpi=100)

        with open('media/model.pkl', 'wb') as f:
            pickle.dump(self.model, f)

        return acc, prec, rec, f1
