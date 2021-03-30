from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ElasticNet
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix


class ENAdvAttack:

    def __init__(self, xTrain, yTrain, yTest, xTest, model):
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.yTest = yTest
        self.xTest = xTest
        self.model = model
        self.dataAmount = 100

    def generate_attack(self):
        art_classifier = SklearnClassifier(model=self.model)

        en = ElasticNet(classifier=art_classifier, confidence=0.0, targeted=False, learning_rate=1e-1, max_iter=30,
                        binary_search_steps=20, initial_const=1e-3, batch_size=1)

        print("Start ENAdvAttack attack")
        x_train_adv = en.generate(self.xTrain[:self.dataAmount])

        score_train = self.model.score(x_train_adv, self.yTrain[:self.dataAmount])
        print("Adversarial Training Score: %.4f" % score_train)

        x_test_adv = en.generate(self.xTest[:self.dataAmount])

        score_test = self.model.score(x_test_adv, self.yTest[:self.dataAmount])
        print("Adversarial Test Score: %.4f" % score_test)

        yPred = self.model.predict(x_test_adv)

        prec = precision_score(self.yTest[:self.dataAmount], yPred)
        print("The precision is {}".format(prec))

        rec = recall_score(self.yTest[:self.dataAmount], yPred)
        print("The recall is {}".format(rec))

        f1 = f1_score(self.yTest[:self.dataAmount], yPred)
        print("The F1-Score is {}".format(f1))

        y_pred_proba = self.model.predict_proba(x_test_adv)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(self.yTest[:self.dataAmount], y_pred_proba)

        plt.figure(figsize=(4, 4))
        plt.plot([0, 1], [0, 1], linestyle='--')
        # plot the roc curve for the model
        plt.plot(fpr, tpr, marker='.', label='ROC Curve')
        plt.title('ROC Curve')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('media/a_env_roc.png', dpi=100)
        plt.legend()

        LABELS = ['Normal', 'Fraud']
        conf_matrix = confusion_matrix(self.yTest[:self.dataAmount], yPred)
        plt.figure(figsize=(4, 4))
        sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
        plt.title("Confusion matrix")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.savefig('media/a_env_confusion_matrix.png', dpi=100)

        return score_train, score_test, x_train_adv, x_test_adv, prec, rec, f1
