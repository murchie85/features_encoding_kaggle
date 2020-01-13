import pandas as pd
from sklearn import preprocessing
import lightgbm as lgb
from sklearn import metrics
import get_split
import get_train

clicks = pd.read_parquet('baseline_data.pqt')
print('printing head')
print(clicks.head(10))


# COUNT ENCODINGS  0.965 ACCURACY

import category_encoders as ce

cat_features = ['ip', 'app', 'device', 'os', 'channel']
train, valid, test = get_split.get_data_splits(clicks)

# Create the count encoder
count_enc = ce.CountEncoder(cols=cat_features)


# Learn encoding from the training set
count_enc.fit(train[cat_features])

# Apply encoding to the train and validation sets as new columns
# Make sure to add `_count` as a suffix to the new columns

train_encoded = train.join(count_enc.transform(train[cat_features]).add_suffix('_count'))
valid_encoded = valid.join(count_enc.transform(valid[cat_features]).add_suffix('_count'))


# Train the model on the encoded datasets
# This can take around 30 seconds to complete
_ = get_train.train_model(train_encoded, valid_encoded)


