import pandas as pd
import numpy as np

# load the data
train_df = pd.read_csv("../SantanderData/train.csv")
test_df = pd.read_csv("../SantanderData/test.csv")

# get the feature and target
features = train_df.columns[2:]
train_X = train_df[features].values
train_Y = train_df['target'].values
test_X = test_df[features].values
# load the model
from Models import lgbModel,xgbModel,cbModel

# setting the params
nfold = 5
param = {
        'num_leaves': 10,
        'max_bin': 119,
        'min_data_in_leaf': 11,
        'learning_rate': 0.02,
        'min_sum_hessian_in_leaf': 0.00245,
        'bagging_fraction': 1.0,
        'bagging_freq': 5,
        'feature_fraction': 0.05,
        'lambda_l1': 4.972,
        'lambda_l2': 2.276,
        'min_gain_to_split': 0.65,
        'max_depth': 14,
        'save_binary': True,
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0
}
lgbModel = lgbModel(train_X,train_Y,test_X,nfold,param,features)
CVScore_lgb,feature_importance_df,predictions_lgb = lgbModel.train_model()
xgbModel = xgbModel(train_X,train_Y,test_X,nfold,param,features)
CVScore_xgb,predictions_xgb = xgbModel.train_model()
cbModel = cbModel(train_X,train_Y,test_X,nfold,param,features)
CVScore_cb,predictions_cb = xgbModel.train_model()

# output the cvscore
print("\n\nlgb model CV AUC: {:<0.5f}".format(CVScore_lgb))
print("\n\nxgb model CV AUC: {:<0.5f}".format(CVScore_xgb))
print("\n\ncb model CV AUC: {:<0.5f}".format(CVScore_cb))

# plot the feature importance
from plot_feature_importance import feature_barplot
feature_barplot(feature_importance_df)

#output the prediction
sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})
sub_df["target"] = predictions_lgb
sub_df.to_csv("../SantanderData/submission_default_feature_lgb.csv", index=False)
sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})
sub_df["target"] = predictions_xgb
sub_df.to_csv("../SantanderData/submission_default_feature_xgb.csv", index=False)
sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})
sub_df["target"] = predictions_cb
sub_df.to_csv("../SantanderData/submission_default_feature_cb.csv", index=False)