import pandas as pd
from sklearn import preprocessing
import lightgbm as lgb
from sklearn import metrics
import get_split
import get_train

#click_data = pd.read_csv('train_sample.csv', parse_dates=['click_time'])
clicks = pd.read_parquet('baseline_data.pqt')

print('printing head')
print(clicks.head(10))



# BASELINE WIHT NO ENCODINGS (UNCOMENT FOR BASIC 0.962 ACCURACY)
print("Baseline model")
train, valid, test = get_split.get_data_splits(clicks)
_ = get_train.train_model(train, valid)




