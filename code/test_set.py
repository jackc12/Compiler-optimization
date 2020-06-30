import pandas as pd
import torch
test_set = torch.tensor(pd.read_csv('../data/test.csv').values, dtype=torch.float)