import math
import pandas as pd
import numpy as np
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

class lgbModel(object):
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
        feature_importance_df = pd.DataFrame()
        i = 1
        for train_index, valid_index in skf.split(self.train_X, self.train_Y):
            print("\nfold {}".format(i))

            xg_train = lgb.Dataset(self.train_X[train_index, :],
                                   label=self.train_Y[train_index],
                                   free_raw_data=False
                                   )
            xg_valid = lgb.Dataset(self.train_X[valid_index, :],
                                   label=self.train_Y[valid_index],
                                   free_raw_data=False
                                   )
            evals_result = {} #record the result for metric plotting
            clf = lgb.train(self.params, xg_train, 15000, valid_sets=[xg_train,xg_valid], verbose_eval=1000, early_stopping_rounds=250,evals_result=evals_result)
            plt.figure(figsize=(12, 6))
            lgb.plot_metric(evals_result, metric='auc')
            plt.title("Metric")
            plt.savefig("figure/Metric{}.png".format(i))
            oof[valid_index] = clf.predict(self.train_X[valid_index, :], num_iteration=clf.best_iteration)
            fold_importance_df = pd.DataFrame()
            fold_importance_df["Feature"] = self.features
            fold_importance_df["importance"] = clf.feature_importance()
            fold_importance_df["fold"] = i
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            predictions += clf.predict(self.test_X, num_iteration=clf.best_iteration) / self.nfold
            i = i + 1
        CVScore = roc_auc_score(self.train_Y, oof)
        return CVScore,feature_importance_df,predictions

class xgbModel(object):
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
            clf = xgb.XGBClassifier(max_depth=2,
                                      n_estimators=2000,
                                      colsample_bytree=0.3,
                                      learning_rate=0.02,
                                      objective='binary:logistic',
                                      n_jobs=-1)

            clf.fit(self.train_X[train_index,:], self.train_Y[train_index],
                      eval_set=[(self.train_X[train_index,:], self.train_Y[train_index])],
                      verbose=False,
                      early_stopping_rounds=1000)
            oof[valid_index] = clf.predict(self.train_X[valid_index, :])
            print("fold{:}'s roc is: {:}".format(i,roc_auc_score(oof[valid_index],self.train_Y[valid_index])))
            predictions += clf.predict(self.test_X) / self.nfold
            i = i + 1
        CVScore = roc_auc_score(self.train_Y, oof)
        return CVScore,predictions
class cbModel(object):
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
            clf = cb.CatBoostClassifier(iterations=5000,
                                          max_depth=2,
                                          learning_rate=0.02,
                                          colsample_bylevel=0.03,
                                          objective="Logloss")

            clf.fit(self.train_X[train_index,:], self.train_Y[train_index],
                      eval_set=[(self.train_X[train_index,:], self.train_Y[train_index])],
                      verbose=False,
                      early_stopping_rounds=1000)
            oof[valid_index] = clf.predict(self.train_X[valid_index, :])
            print("fold{:}'s roc is: {:}".format(i,roc_auc_score(oof[valid_index],self.train_Y[valid_index])))
            predictions += clf.predict(self.test_X) / self.nfold
            i = i + 1
        CVScore = roc_auc_score(self.train_Y, oof)
        return CVScore,predictions