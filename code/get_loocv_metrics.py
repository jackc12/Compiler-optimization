import pandas as pd
from find_token import find_token

with open('../leave_one_out_epochs=50_batch_size=4', 'r') as b_4:
    b_4_lines = b_4.readlines()

pd.DataFrame({
    'App Name': [find_token(line, '', ', epoch') for line in b_4_lines],
    'Train Loss': [float(find_token(line, 'train loss: ', ', test loss: ')) for line in b_4_lines],
    'Test Loss': [float(find_token(line, 'test loss: ', ', average test_loss ')) for line in b_4_lines],
    'Average Test Loss': [float(find_token(line, 'average test_loss ', '\n')) for line in b_4_lines],
}).to_csv(path_or_buf='./data/metrics.csv')