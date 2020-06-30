import torch, pandas as pd, numpy as np

from net import net
from train_set import train_set
from test_set import test_set
from predict import predict

apps = pd.read_csv('../data/data.csv').iloc[:,0]#.unique()

cobayn_speedups = dict(zip(apps.unique(), ([i] for i in [1.05, 1.4, 1.30, 1.08, 1.12, 1.15, 1.6, 1.56, 1.54, 2.3, 1.25, 1.48, 1.23, 2.15, 1.40, 1.53, 1.37, 1.72, 1.38, 2.35, 2.85, 2.14, 1.23, 1.23])))

cnn_speedups = {}
data = torch.cat([train_set[:2304], test_set, train_set[2304:]])

for i in range(0,3072,128):
    shuffled_data = data[i:i+128,:][torch.randperm(128)]
    train_predictions, optimizations = predict(net, 32, shuffled_data), shuffled_data
    a = pd.DataFrame(shuffled_data.numpy())
    prediction = train_predictions.sum(axis=1).min()
    o2 = a[(a[0] == 0) & (a[1] == 0) & (a[2] == 0) & (a[3] == 0) & (a[4] == 0) & (a[5] == 0) & (a[6] == 1)].iloc[:,-5:].sum(axis=1).values[0]
    cnn_speedups[apps[i]] = [prediction/o2]

pd.DataFrame(cobayn_speedups).to_csv(path_or_buf='../data/cobayn_speedups.csv')
pd.DataFrame(cnn_speedups).to_csv(path_or_buf='../data/cnn_speedups.csv')

