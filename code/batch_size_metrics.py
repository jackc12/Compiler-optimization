import pandas as pd

batch_size_metrics = pd.read_csv(filepath_or_buffer='../data/batch_size_metrics.csv').drop('Unnamed: 0', axis=1)
