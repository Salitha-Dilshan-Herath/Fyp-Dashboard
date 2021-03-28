from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

class LoRegression:

    def __init__(self, xTrain, yTrain, yTest, xTest):
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.yTest = yTest
        self.xTest = xTest

    def model_train(self):

        # random forest model creation
        rfc = LogisticRegression()
        rfc.fit(self.xTrain, self.yTrain)
        #predictions

        print("start predication")
        yPred = rfc.predict(self.xTest)

        print("The model used is Logistic Regression classifier")

        acc = accuracy_score(self.yTest, yPred)
        print("The accuracy is {}".format(acc))

        prec = precision_score(self.yTest, yPred)
        print("The precision is {}".format(prec))

        rec = recall_score(self.yTest, yPred)
        print("The recall is {}".format(rec))

        f1 = f1_score(self.yTest, yPred)
        print("The F1-Score is {}".format(f1))

        MCC = matthews_corrcoef(self.yTest, yPred)
        print("The Matthews correlation coefficient is {}".format(MCC))

        y_pred_proba = rfc.predict_proba(self.xTest)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(self.yTest, y_pred_proba)

        plt.figure(figsize=(4, 4))
        plt.plot([0, 1], [0, 1], linestyle='--')
        # plot the roc curve for the model
        plt.plot(fpr, tpr, marker='.', label='ROC Curve')
        plt.title('ROC Curve')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('media/normal_env_roc.png', dpi=100)
        plt.legend()

        LABELS = ['Normal', 'Fraud']
        conf_matrix = confusion_matrix(self.yTest, yPred)
        plt.figure(figsize=(4, 4))
        sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
        plt.title("Confusion matrix")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.savefig('media/normal_env_confusion_matrix.png', dpi=100)

        return rfc, acc, prec, rec, f1