import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#Load the data
data = pd.read_csv('../data/data.csv')
relevent_data = data.drop('code_size', axis=1)
test_app = 'consumer_tiffmedian'


from net import net
losses = []
net = net.to(device)
net.train()


def train_test_split(test_app):
  train_data  = relevent_data[relevent_data['APP_NAME'] != test_app]
  test_data = relevent_data[relevent_data['APP_NAME'] == test_app]
  scaler = MinMaxScaler(feature_range=(0,1)).fit(relevent_data[relevent_data['APP_NAME'] != test_app].iloc[:,1:-5])
  scaled_train_predictors = scaler.transform(relevent_data[relevent_data['APP_NAME'] != test_app].iloc[:,1:-5])
  train_targets = relevent_data[relevent_data['APP_NAME'] != test_app].iloc[:,-5:]
  scaled_train_targets = []
  for app in range(0, train_targets.shape[0], 128):
    scaled_train_targets.append(MinMaxScaler(feature_range=(0,1)).fit_transform(train_targets.iloc[app:app+128]))
  scaled_train_targets = np.array(scaled_train_targets).reshape(train_targets.shape[0],5)
  scaled_test_predictors = MinMaxScaler(feature_range=(0,1)).fit_transform(relevent_data[relevent_data['APP_NAME'] == test_app].iloc[:,1:-5])
  scaled_test_targets = MinMaxScaler(feature_range=(0,1)).fit_transform(relevent_data[relevent_data['APP_NAME'] == test_app].iloc[:,-5:])
  train_set = torch.tensor(
      data=np.concatenate(
          (
              scaled_train_predictors,
              scaled_train_targets
          ),
          axis=1
      ),
      dtype=torch.float
      ).to(device)
  test_set = torch.tensor(
    data=np.concatenate(
        (
            scaled_test_predictors,
            scaled_test_targets
        ),
        axis=1
    ),
    dtype=torch.float
  ).to(device)
  return train_set, test_set


#Set hyperparameters
batch_size = 32
epochs = 50
lr=1e-4
weight_decay = 1e-4

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)


for test_app in relevent_data['APP_NAME'].unique():
  train_set, test_set = train_test_split(test_app)
  test_losses = []
  #Training loop
  for epoch in range(epochs):
      
      #Set train loss to zero
      running_loss = 0.0
      
      #Shuffle training set
      shuffled_train_set = train_set[torch.randperm(train_set.shape[0])]
      
      for start_index in range(0, shuffled_train_set.shape[0] - batch_size, batch_size):
          #get batches
          X = shuffled_train_set[start_index:start_index + batch_size, :-5].view(batch_size,1,285)
          y = shuffled_train_set[start_index:start_index + batch_size, -5:]
          
          #zero gradients
          optimizer.zero_grad()
          
          #forward pass
          outputs = net(X).view(batch_size,5)
          
          #get loss
          loss = criterion(outputs, y)
          
          #backward pass
          loss.backward()
          optimizer.step()
          running_loss += loss.item()

      #Set test loss to zero
      test_loss = 0.0

      #Shuffle test set
      shuffled_test_set = test_set[torch.randperm(test_set.shape[0])]

      for start_index in range(0, shuffled_test_set.shape[0] - batch_size, batch_size):

          #get batches
          X = shuffled_test_set[start_index:start_index + batch_size, :-5].view(batch_size,1,285)
          y = shuffled_test_set[start_index:start_index + batch_size, -5:]

          #forward pass
          outputs = net(X).view(batch_size,5)

          #get loss
          loss = criterion(outputs, y)
          test_loss += loss.item() * 23
      test_losses.append(test_loss)

  print('{}, epoch {}, train loss: {}, test loss: {}, average test_loss {}'.format(test_app, epoch, running_loss, test_loss, sum(test_losses)/len(test_losses) ))


from sklearn.metrics import mean_squared_error

from predict import predict
from plot_execution_times import plot_execution_times


predictions = predict(net, 32, test_set)

print('Test Loss:', mean_squared_error(test_set[:,-5:], predictions))
plot_execution_times(predictions, test_set[:, -5:])