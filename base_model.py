from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

class lrmodel(object):
    def __init__(self,train_X,train_Y,test_X,nfold,params,features):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.nfold = nfold
        self.params = params
        self.features = features

    def train_model(self):
        skf = StratifiedKFold(n_splits=self.nfold, shuffle=True, random_state=2019)
        oof = np.zeros(len(self.train_X))
        predictions = np.zeros(len(self.test_X))
        i = 1
        for train_index, valid_index in skf.split(self.train_X, self.train_Y):
            print("\nfold {}".format(i))
            clf = GaussianNB()
            clf.fit(self.train_X[train_index,:],self.train_Y[train_index])
            oof[valid_index] = clf.predict(self.train_X[valid_index, :])
            print("fold:{:1},roc score:{:2}".format(i,roc_auc_score(oof[valid_index],self.train_Y[valid_index])),)
            predictions += clf.predict(self.test_X)/self.nfold
            i = i + 1
        CVScore = roc_auc_score(self.train_Y, oof)
        return CVScore,predictions