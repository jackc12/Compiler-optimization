import pandas as pd
import torch
test_set_unscaled_ys = torch.tensor(pd.read_csv('../data/test_unscaled_ys.csv').values, dtype=torch.float)[torch.randperm(128)]