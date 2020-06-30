import pandas as pd, numpy as np, matplotlib.pyplot as plt
apps = pd.read_csv('../data/data.csv')['APP_NAME'].unique()

from cnn_speedups import cnn_speedups
from cobayn_speedups import cobayn_speedups

def plot_speedups():
	pd.DataFrame({
	    'application': np.arange(24),
	    'cobayn': cobayn_speedups,
	    'cnn': cnn_speedups
	}).plot.bar(x='application', y=['cobayn', 'cnn'], color=['r','b'], figsize=(15,4), ylim=(0,3.5)).set(title='speedup w.r.t. O2')
	plt.xticks(np.arange(24), labels=apps, rotation=80)

plot_speedups()