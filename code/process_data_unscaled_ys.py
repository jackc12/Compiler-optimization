import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('../data/data.csv')
relevent_data = data.drop('code_size', axis=1)
test_app = 'consumer_tiffmedian'

def train_test_split(test_app):
  train_data  = relevent_data[relevent_data['APP_NAME'] != test_app].iloc[:,1:].values
  test_data = relevent_data[relevent_data['APP_NAME'] == test_app].iloc[:,1:].values
  scaler = MinMaxScaler(feature_range=(0,1)).fit(train_data)
  return scaler.transform(train_data), scaler.transform(test_data)

train_set, test_set = train_test_split(test_app)
pd.DataFrame(test_set).to_csv(path_or_buf='../processed_data/test_unscaled_ys.csv', index=False)