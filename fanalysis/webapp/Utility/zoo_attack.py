from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt_curve
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix


class ZooAdvAttack:

    def __init__(self, xDataTrain, yDataTrain, yDataTest, xDataTest, model):
        self.xDataTrain = xDataTrain
        self.yDataTrain = yDataTrain
        self.yDataTest = yDataTest
        self.xDataTest = xDataTest
        self.model = model
        self.dataAmount = 100

    def generate_attack(self):
        art_classifier = SklearnClassifier(model=self.model)

        zoo = ZooAttack(classifier=art_classifier, confidence=0.0, targeted=False, learning_rate=1e-1, max_iter=30,
                        binary_search_steps=20, initial_const=1e-3, abort_early=True, use_resize=False,
                        use_importance=False, nb_parallel=1, batch_size=1, variable_h=0.25)

        print("Start attack")
        x_train_adv = zoo.generate(self.xDataTrain[:self.dataAmount])

        score_train = self.model.score(x_train_adv, self.yDataTrain[:self.dataAmount])
        print("Adversarial Training Score: %.4f" % score_train)

        x_test_adv = zoo.generate(self.xDataTest[:self.dataAmount])

        score_test = self.model.score(x_test_adv, self.yDataTest[:self.dataAmount])
        print("Adversarial Test Score: %.4f" % score_test)

        yRPred = self.model.predict(x_test_adv)

        prec = precision_score(self.yDataTest[:self.dataAmount], yRPred)
        print("precision value is {}".format(prec))

        rec = recall_score(self.yDataTest[:self.dataAmount], yRPred)
        print("recall value is {}".format(rec))

        f1 = f1_score(self.yDataTest[:self.dataAmount], yRPred)
        print("F1-Score value is {}".format(f1))

        y_pred_proba = self.model.predict_proba(x_test_adv)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(self.yDataTest[:self.dataAmount], y_pred_proba)

        plt_curve.figure(figsize=(4, 4))
        plt_curve.plot([0, 1], [0, 1], linestyle='--')
        # genarate the roc_curve for the model
        plt_curve.plot(fpr, tpr, marker='.', label='ROC Curve')
        plt_curve.title('ROC Curve')
        plt_curve.ylabel('True Positive Rate')
        plt_curve.xlabel('False Positive Rate')
        plt_curve.savefig('media/a_env_roc.png', dpi=100)
        plt_curve.legend()

        axiesLables = ['Normal', 'Fraud']
        conf_matrix = confusion_matrix(self.yDataTest[:self.dataAmount], yRPred)
        plt_curve.figure(figsize=(4, 4))
        sns.heatmap(conf_matrix, xticklabels=axiesLables, yticklabels=axiesLables, annot=True, fmt="d");
        plt_curve.title("Confusion matrix")
        plt_curve.ylabel('True class')
        plt_curve.xlabel('Predicted class')
        plt_curve.savefig('media/a_env_confusion_matrix.png', dpi=100)

        return score_train, score_test, x_train_adv, x_test_adv, prec, rec, f1
