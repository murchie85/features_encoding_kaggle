import pandas as pd
from sklearn import preprocessing
import lightgbm as lgb
from sklearn import metrics
import category_encoders as ce
import get_split
import get_train

clicks = pd.read_parquet('baseline_data.pqt')
print('printing head')
print(clicks.head(10))


#cat_features = ['ip', 'app', 'device', 'os', 'channel']
cat_features = ['app', 'device', 'os', 'channel']

train, valid, test = get_split.get_data_splits(clicks)

# Create the CatBoost encoder
# Have to tell it which features are categorical when they aren't strings
cb_enc = ce.CatBoostEncoder(cols=cat_features, random_state=7)

# Learn encoding from the training set
# train[cat_features] = the training split
# train['is_attributed'] = target
cb_enc.fit(train[cat_features], train['is_attributed'])

# Apply encoding to the train and validation sets as new columns
# Make sure to add `_cb` as a suffix to the new columns
train_encoded = train.join(cb_enc.transform(train[cat_features]).add_suffix('_cb'))
valid_encoded = valid.join(cb_enc.transform(valid[cat_features]).add_suffix('_cb'))

_ = get_train.train_model(train, valid)


"""
Note removing IP is better because 
Target encoding attempts to measure the population mean of the target for each level in a categorical feature. 
This means when there is less data per level, the estimated mean will be further away from the "true" mean, 
there will be more variance. 
There is little data per IP address so it's likely that the estimates are much NOISER than for the other features. 
The model will rely heavily on this feature since it is extremely predictive. This causes it to make fewer splits on other features, 
and those features are fit on just the errors left over accounting for IP address. So, the model will 
perform very poorly when seeing new IP addresses that weren't in the training data (which is likely most new data). 
Going forward, we'll leave out the IP feature when trying different encodings.

"""



# encodes four cats then appends columns to full set
encoded = cb_enc.transform(clicks[cat_features])
for col in encoded:
    clicks.insert(len(clicks.columns), col + '_cb', encoded[col])

# clicks.insert(col_number, colname+_cb, actual encoded val)
"""
encoded looks like 
	app	device	os	channel
0	0.028329	0.152087	0.138712	0.034049
1	0.995828	0.152087	0.138712	0.950244


clicks now has encoded four columns added 

"""
