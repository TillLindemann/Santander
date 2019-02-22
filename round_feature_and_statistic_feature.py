import pandas as pd
import numpy as np

# load the data
train_df = pd.read_csv("../SantanderData/train.csv")
test_df = pd.read_csv("../SantanderData/test.csv")

# adding the statistic feature
i = 1
for df in [test_df, train_df]:
    idx = df.columns.values[i:i+200]
    df['sum'] = df[idx].sum(axis=1)
    df['min'] = df[idx].min(axis=1)
    df['max'] = df[idx].max(axis=1)
    df['mean'] = df[idx].mean(axis=1)
    df['std'] = df[idx].std(axis=1)
    df['skew'] = df[idx].skew(axis=1)
    df['kurt'] = df[idx].kurtosis(axis=1)
    df['med'] = df[idx].median(axis=1)
    i = i + 1

# adding the round feature
features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
for feature in features:
    train_df['r3_'+feature] = np.round(train_df[feature], 3)
    test_df['r3_'+feature] = np.round(test_df[feature], 3)
    train_df['r2_'+feature] = np.round(train_df[feature], 2)
    test_df['r2_'+feature] = np.round(test_df[feature], 2)
    train_df['r1_'+feature] = np.round(train_df[feature], 1)
    test_df['r1_'+feature] = np.round(test_df[feature], 1)

# get ready for model training
features = train_df.columns.values[2:]
print(features)
train_X = train_df[features].values
train_Y = train_df['target'].values
test_X = test_df[features].values

# train model
# load the model
from Models import lgbModel

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
CVScore,feature_importance_df,predictions = lgbModel.train_model()

# output the cvscore
print("\n\nCV AUC: {:<0.5f}".format(CVScore))

# plot the feature importance
from plot_feature_importance import feature_barplot
feature_barplot(feature_importance_df)

#output the prediction
sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})
sub_df["target"] = predictions
sub_df.to_csv("../SantanderData/submission_round_feature_and_statistic_feature.csv", index=False)