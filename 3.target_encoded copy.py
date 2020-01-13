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


cat_features = ['ip', 'app', 'device', 'os', 'channel']
train, valid, test = get_split.get_data_splits(clicks)

# Create the target encoder. You can find this easily by using tab completion.
# Start typing ce. the press Tab to bring up a list of classes and functions.
target_enc = ce.TargetEncoder(cols=cat_features)

# Learn encoding from the training set. Use the 'is_attributed' column as the target.
target_enc.fit(train[cat_features], train['is_attributed'])

# Apply encoding to the train and validation sets as new columns
# Make sure to add `_target` as a suffix to the new columns
train_encoded = train.join(target_enc.transform(train[cat_features]).add_suffix('_target'))
valid_encoded = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_target'))

_ = get_train.train_model(train_encoded, valid_encoded)

