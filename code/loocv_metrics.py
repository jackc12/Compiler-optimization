import pandas as pd

loocv_metrics = pd.read_csv(filepath_or_buffer='../data/loocv_metrics.csv').drop('Unnamed: 0', axis=1)