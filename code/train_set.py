import pandas as pd
import torch
train_set = torch.tensor(pd.read_csv('../data/train.csv').values, dtype=torch.float)