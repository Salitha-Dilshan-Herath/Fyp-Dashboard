import numpy as np
import pandas as pd
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

from .dp_attack import DPAdvAttack
from .en_attack import ENAdvAttack
from .zoo_attack import ZooAdvAttack

from .random_forest import RandomForest
from .lo_regression import LoRegression
from .d_tree import DTree

from .adversarial_defence import AdversarialDefence


class DataPreprocessing:

    df = pd.DataFrame()
    dataPercentage = 0.1

    selectedParameter = 'Class'
    selectedTrainingModel = "rf"
    selectedAttackType = "zoo"
    selectedDefenceType = "adv_train"

    trainModel = ""
    X_train_var, X_test_var, yTrain, yTest, x_train_adv, x_test_adv = [], [], [], [], [], []

    def __init__(self, url):
        self.url = url

    def read_file(self):
        self.df = pd.read_csv(os.getcwd() + self.url)

        return list(self.df.columns)

    def filter_data(self):

        print(self.dataPercentage, self.selectedTrainingModel, self.selectedParameter)
        self.df = self.df.sample(frac=self.dataPercentage)

    def data_frame_describe(self):
        print("data_frame_describe")

        fraud = self.df[self.df[self.selectedParameter] == 1]
        valid = self.df[self.df[self.selectedParameter] == 0]
        outlier_fraction = len(fraud) / float(len(valid))
        fraud_case = 'Fraud Cases: {}'.format(len(self.df[self.df[self.selectedParameter] == 1]))
        valid_case = 'Valid Transactions: {}'.format(len(self.df[self.df[self.selectedParameter] == 0]))
        return outlier_fraction, fraud_case, valid_case

    def implement_feature_selection(self):

        print("implement_feature_selection")
        X = self.df.drop([self.selectedParameter], axis=1)
        Y = self.df[self.selectedParameter]

        # getting just the values for the sake of processing
        # (its a numpy array with no columns)
        xData = X.values
        yData = Y.values

        sm = SMOTE(k_neighbors=2)

        X_train_over, y_train_over = sm.fit_resample(xData, yData)

        # split the data into training and testing sets
        xTrain, xTest, yTrain, yTest = train_test_split(X_train_over, y_train_over, test_size=0.2, random_state=42)

        var = VarianceThreshold(threshold=.5)
        var.fit(xTrain, yTrain)
        X_train_var = var.transform(xTrain)
        X_test_var = var.transform(xTest)

        self.X_train_var = X_train_var
        self.X_test_var = X_test_var
        self.yTest = yTest
        self.yTrain = yTrain

    def train_model(self):

        print("call train", self.selectedTrainingModel)

        if self.selectedTrainingModel == "rf":

            print("rf call")
            rf_obj = RandomForest(self.X_train_var, self.yTrain, self.yTest, self.X_test_var)
            rfc, acc, prec, rec, f1 = rf_obj.model_train()
            self.trainModel = rfc

            return rfc, acc, prec, rec, f1

        elif self.selectedTrainingModel == "lr":
            rf_obj = LoRegression(self.X_train_var, self.yTrain, self.yTest, self.X_test_var)
            rfc, acc, prec, rec, f1 = rf_obj.model_train()
            self.trainModel = rfc

            return rfc, acc, prec, rec, f1

        elif self.selectedTrainingModel == "dt":
            rf_obj = DTree(self.X_train_var, self.yTrain, self.yTest, self.X_test_var)
            rfc, acc, prec, rec, f1 = rf_obj.model_train()
            self.trainModel = rfc

            return rfc, acc, prec, rec, f1

    def attack(self):

        if self.selectedAttackType == "zoo":
            at_obj = ZooAdvAttack(self.X_train_var, self.yTrain, self.yTest, self.X_test_var, self.trainModel)
            score_train, score_test, self.x_train_adv, self.x_test_adv, prec, rec, f1 = at_obj.generate_attack()
            return score_train, score_test, prec, rec, f1

        elif self.selectedAttackType == "en":
            at_obj = ENAdvAttack(self.X_train_var, self.yTrain, self.yTest, self.X_test_var, self.trainModel)
            score_train, score_test, self.x_train_adv, self.x_test_adv, prec, rec, f1 = at_obj.generate_attack()
            return score_train, score_test, prec, rec, f1

        elif self.selectedAttackType == "dp":
            at_obj = DPAdvAttack(self.X_train_var, self.yTrain, self.yTest, self.X_test_var, self.trainModel)
            score_train, score_test, self.x_train_adv, self.x_test_adv, prec, rec, f1 = at_obj.generate_attack()
            return score_train, score_test, prec, rec, f1

    def defence(self):

        if self.selectedDefenceType == "adv_train":
            df_obj = AdversarialDefence(self.X_train_var, self.x_train_adv, self.x_test_adv, self.yTrain, self.yTest,
                                        self.trainModel)
            acc, prec, rec, f1 = df_obj.defence()

            return acc, prec, rec, f1
