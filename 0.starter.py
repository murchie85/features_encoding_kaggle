import pandas as pd
from sklearn import preprocessing
import lightgbm as lgb
from sklearn import metrics

click_data = pd.read_csv('train_sample.csv',
                         parse_dates=['click_time'])
print('printing head')
print(click_data.head(10))


# Add new columns for timestamp features day, hour, minute, and second
clicks = click_data.copy()
click_times = click_data['click_time']

# slice up click_times and reassign to clicks as new columns
clicks['day'] = click_times.dt.day.astype('uint8')
clicks['hour'] = click_times.dt.hour.astype('uint8')
clicks['minute'] = click_times.dt.minute.astype('uint8')
clicks['second'] = click_times.dt.second.astype('uint8')



cat_features = ['ip', 'app', 'device', 'os', 'channel']

# Create new labells columns (_labels) in clicks using preprocessing.LabelEncoder()
label_encoder = preprocessing.LabelEncoder()
for feature in cat_features:
    encoded = label_encoder.fit_transform(clicks[feature])
    clicks[feature + '_labels'] = encoded


print('printing head')
print(clicks.head())

"""
The ip column has 58,000 values, which means it will create an extremely sparse matrix with 58,000 columns. 
This many columns will make your model run very slow, so in general you want to avoid one-hot encoding features with many levels. 
LightGBM models work with label encoded features, so you don't actually need to one-hot encode the categorical features.
"""



# QUESTION
"""
This is time series data. Are they any special considerations when creating train/test splits for time series? If so, what and why?
"""

"""
Since our model is meant to predict events in the future, we must also validate the model on events in the future. 
If the data is mixed up between the training and test sets, 
then future data will leak in to the model and our validation results will overestimate the performance on new data.
"""

"""
Here we'll create training, validation, and test splits. First, `clicks` DataFrame is sorted in order of increasing time. 
The first 80% of the rows are the train set, the next 10% are the validation set, and the last 10% are the test set.
"""


feature_cols = ['day', 'hour', 'minute', 'second', 
                'ip_labels', 'app_labels', 'device_labels',
                'os_labels', 'channel_labels']

valid_fraction = 0.1
clicks_srt = clicks.sort_values('click_time')
valid_rows_size = int(len(clicks_srt) * valid_fraction)


# from beginning up to the last 0.1 (valid_rows_size) * 2

train = clicks_srt[:-valid_rows_size * 2] # first 80%
valid = clicks_srt[-valid_rows_size * 2:-valid_rows_size] # second last part from the end 
test = clicks_srt[-valid_rows_size:] # last part from the end


# Train with LightGBM lgb

dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])
dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])
dtest = lgb.Dataset(test[feature_cols], label=test['is_attributed'])

param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 1000
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10)


# Evaluate the model


ypred = bst.predict(test[feature_cols])
score = metrics.roc_auc_score(test['is_attributed'], ypred)
print(f"Test score: {score}")
