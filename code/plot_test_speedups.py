import pandas as pd, numpy as np, matplotlib.pyplot as plt
apps = pd.read_csv('../data/data.csv')['APP_NAME'].unique()

from cnn_speedups import cnn_speedups
from cobayn_speedups import cobayn_speedups

def plot_test_speedups():
	pd.DataFrame({
	    'application': [0],
	    'cobayn': cobayn_speedups[-6],
	    'cnn': cnn_speedups[-6]
	}).plot.bar(x='application', y=['cobayn', 'cnn'], color=['r','b'], figsize=(5,4), ylim=(0,2)).set(title='speedup w.r.t. O2')
	plt.xticks([0], labels=['consumer_tiffmedian'], rotation=80)

plot_test_speedups()