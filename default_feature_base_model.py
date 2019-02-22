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

from base_model import lrmodel
nfold = 5
params = {
    'penalty':'l2',
    'class_weight':'balanced',
    'random_state':2019,
    'solver':'sag'
}

lrmodel = lrmodel(train_X,train_Y,test_X,nfold,params,features)
CVscore,predictions = lrmodel.train_model()

print("\n\n CV Score:{:<0.5f}".format(CVscore))

sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})
sub_df["target"] = predictions
sub_df.to_csv("../SantanderData/submission_default_feature_base_model.csv", index=False)